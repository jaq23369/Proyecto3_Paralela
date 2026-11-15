"""Perfila un paso de entrenamiento en GPU y lista los kernels CUDA reales (nombre, grid, bloque).

Es la VERIFICACIÓN empírica de la tabla de mapeo hilos/bloques/grids de
``docs/04_diseno_algoritmo_paralelo.md``: cuDNN/cuBLAS eligen en ejecución qué kernels lanzar
y con qué geometría, y ``torch.profiler`` (CUPTI) la registra.

Uso::  python extras/profile_kernels.py --tag pc1 --model cnn2
Salida: results/kernel_profile_<tag>_<model>.csv  (+ resumen por consola)
"""

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="cnn2", choices=["cnn1", "cnn2", "mlp"])
    ap.add_argument("--steps", type=int, default=20, help="pasos de entrenamiento perfilados")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from torch import nn
    from torch.profiler import ProfilerActivity, profile

    from comun import config
    from paralelo_cuda.modelos import build_model

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    spec = config.MODEL_SPECS[args.model]
    model = build_model(args.model, **spec["kwargs"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"])
    crit = nn.CrossEntropyLoss()
    x = torch.rand(config.BATCH_SIZE, 3, 64, 64, device=device)
    y = torch.randint(0, config.NUM_CLASSES, (config.BATCH_SIZE,), device=device)

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()

    for _ in range(10):  # calentamiento (autotune de cuDNN)
        step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()

    with tempfile.TemporaryDirectory() as tmp:
        trace = Path(tmp) / "trace.json"
        prof.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text(encoding="utf-8"))["traceEvents"]

    agg = defaultdict(lambda: {"calls": 0, "total_us": 0.0, "grids": Counter(), "blocks": Counter()})
    for e in events:
        if e.get("cat") == "kernel":
            a = agg[e["name"]]
            a["calls"] += 1
            a["total_us"] += e.get("dur", 0.0)
            args_ = e.get("args", {})
            a["grids"][tuple(args_.get("grid", []))] += 1
            a["blocks"][tuple(args_.get("block", []))] += 1

    rows = []
    for name, a in agg.items():
        grid, block = a["grids"].most_common(1)[0][0], a["blocks"].most_common(1)[0][0]
        threads = 1
        for v in (*grid, *block):
            threads *= v
        rows.append({"kernel": name[:110], "llamadas": a["calls"], "tiempo_total_us": round(a["total_us"], 1),
                     "grid": "x".join(map(str, grid)), "bloque": "x".join(map(str, block)),
                     "hilos_totales": threads, "geometrias_distintas": len(a["grids"])})
    df = pd.DataFrame(rows).sort_values("tiempo_total_us", ascending=False).reset_index(drop=True)
    total = df["tiempo_total_us"].sum()
    df["pct_tiempo_gpu"] = (100 * df["tiempo_total_us"] / total).round(1)
    out = config.RESULTS_DIR / f"kernel_profile_{args.tag}_{args.model}.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(f"{len(df)} kernels distintos, {int(df['llamadas'].sum())} lanzamientos en {args.steps} pasos "
          f"({df['llamadas'].sum() / args.steps:.0f} por paso)\n")
    print(df.head(args.top).to_string(index=False))
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
