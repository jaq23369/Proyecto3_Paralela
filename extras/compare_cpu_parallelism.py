"""Escalamiento del preprocesamiento en CPU: procesos vs. hilos, 1..N workers.

Justifica empíricamente la decisión de diseño de ``docs/04_diseno_algoritmo_paralelo.md``
(§4.1) usando ``ProcessPoolExecutor`` y no hilos, y muestra la curva de aceleración
(ley de Amdahl) del preprocesamiento. Mide las 17,400 imágenes del manifiesto.

Uso::  python extras/compare_cpu_parallelism.py --tag pc1 [--reps 3]
Salida: results/cpu_scaling_<tag>.csv  y  results/figures/cpu_scaling_<tag>.png
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def threaded_load(manifest, root, size, workers, chunk=64):
    """Misma tarea que la versión por procesos, pero con hilos (comparación)."""
    import numpy as np

    from preparacion_datos.preprocesamiento import _decode_resize_chunk

    paths = manifest["path"].tolist()
    tasks = [(str(root), paths[i : i + chunk], tuple(size)) for i in range(0, len(paths), chunk)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(_decode_resize_chunk, tasks))
    return np.concatenate(out).astype("float32") / 255.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, nargs="+", default=None)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from comun import config
    from preparacion_datos.preprocesamiento import load_preprocessed_batch, load_preprocessed_parallel

    manifest = pd.read_csv(config.SPLIT_MANIFEST)
    n_cpu = os.cpu_count() or 1
    workers_list = args.workers or sorted({1, 2, 4, max(1, n_cpu // 2), n_cpu})
    rows = []

    def timed(kind, workers, fn):
        for rep in range(1, args.reps + 1):
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
            rows.append({"tag": args.tag, "kind": kind, "workers": workers, "rep": rep, "seconds": dt})
            print(f"  {kind:<10} workers={workers:<3} rep={rep}  {dt:6.2f} s", flush=True)

    print("== Baseline: bucle secuencial del Lab 3")
    timed("secuencial", 1, lambda: load_preprocessed_batch(manifest, config.TRAIN_DIR))
    for w in workers_list:
        print(f"== {w} worker(s)")
        timed("procesos", w, lambda w=w: load_preprocessed_parallel(manifest, config.TRAIN_DIR, workers=w))
        timed("hilos", w, lambda w=w: threaded_load(manifest, config.TRAIN_DIR, config.IMAGE_SIZE, w))

    df = pd.DataFrame(rows)
    out = df.groupby(["kind", "workers"])["seconds"].agg(["mean", "std"]).reset_index()
    base = out[out["kind"] == "secuencial"]["mean"].iloc[0]
    out["speedup_vs_secuencial"] = base / out["mean"]
    out["eficiencia"] = out["speedup_vs_secuencial"] / out["workers"]
    csv = config.RESULTS_DIR / f"cpu_scaling_{args.tag}.csv"
    out.round(4).to_csv(csv, index=False)
    print("\n", out.round(3).to_string(index=False))

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for kind, color in (("procesos", "#e76f51"), ("hilos", "#2a9d8f")):
        s = out[out["kind"] == kind]
        ax.plot(s["workers"], s["speedup_vs_secuencial"], marker="o", label=kind, color=color)
    ax.plot([1, n_cpu], [1, n_cpu], "k--", lw=0.8, label="ideal (lineal)")
    ax.set_xlabel("Workers")
    ax.set_ylabel("Speedup vs. bucle secuencial")
    ax.set_title(f"Preprocesamiento en CPU: procesos vs. hilos — {args.tag}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"cpu_scaling_{args.tag}.png", dpi=150)


if __name__ == "__main__":
    main()
