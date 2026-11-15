"""Entrenamiento COMPLETO (con parada temprana) para validar que el port a PyTorch
conserva la efectividad del Lab 3 y que la GPU no cambia la calidad del modelo.

A diferencia de ``benchmark.py`` (épocas fijas para medir tiempo), aquí se replica el
protocolo del Lab 3: Adam, lote 64, hasta 30 épocas, paciencia 4 sobre el accuracy de
validación y restauración de los mejores pesos. ``test`` se evalúa UNA vez al final.

Referencia del Lab 3 (validación): CNN #1 94.2 %, CNN #2 98.0 %, MLP 34.2 %.

Uso::  python extras/train_full.py --tag pc1 --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--models", nargs="+", default=["cnn1", "cnn2", "mlp"], choices=["cnn1", "cnn2", "mlp"])
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    import torch

    from comun import config
    from paralelo_cuda.modelos import build_model, count_parameters
    from comun.monitor import ResourceMonitor
    from preparacion_datos.preprocesamiento import class_names, encode_labels, load_preprocessed_parallel, to_nchw
    from paralelo_cuda.entrenamiento import evaluate, fit, get_device, set_seed, to_device

    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    manifest = pd.read_csv(config.SPLIT_MANIFEST)
    classes = class_names(manifest)

    with ResourceMonitor(config.LOGS_DIR / f"resources_{args.tag}_trainfull_{args.device}.csv") as mon:
        mon.mark("preprocess")
        t0 = time.perf_counter()
        images, labels = load_preprocessed_parallel(manifest, config.TRAIN_DIR, config.IMAGE_SIZE)
        t_prep = time.perf_counter() - t0
        x, y = to_nchw(images), encode_labels(labels, classes)
        split = manifest["split"].to_numpy()
        data = {s: (to_device(x[split == s], device), to_device(y[split == s], device))
                for s in ("train", "validation", "test")}
        print(f"preprocesamiento: {t_prep:.1f} s | dispositivo: {device}", flush=True)

        rows, histories = [], {}
        for name in args.models:
            spec = config.MODEL_SPECS[name]
            set_seed()
            model = build_model(name, **spec["kwargs"]).to(device)
            print(f"\n=== {name} (lr={spec['lr']}, parámetros={count_parameters(model):,})", flush=True)
            mon.mark(f"train_{name}")
            res = fit(model, *data["train"], *data["validation"], lr=spec["lr"], max_epochs=config.MAX_EPOCHS,
                      patience=config.EARLY_STOPPING_PATIENCE, amp=args.amp)
            test_loss, test_acc = evaluate(model, *data["test"])
            rows.append({"tag": args.tag, "device": args.device, "model": name, "epochs_run": res.epochs_run,
                         "best_epoch": res.best_epoch, "val_acc": round(res.best_val_acc, 4),
                         "test_acc": round(test_acc, 4), "train_seconds": round(res.total_seconds, 1),
                         "sec_per_epoch": round(res.total_seconds / res.epochs_run, 2)})
            histories[name] = res.history
            (config.RESULTS_DIR / "models").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.RESULTS_DIR / "models" / f"{name}_{args.tag}_{args.device}.pt")
            print(f"    -> val={res.best_val_acc:.4f}  test={test_acc:.4f}  ({res.total_seconds:.0f} s)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS_DIR / f"train_full_{args.tag}_{args.device}.csv", index=False)
    (config.RESULTS_DIR / f"train_full_history_{args.tag}_{args.device}.json").write_text(
        json.dumps(histories, indent=1), encoding="utf-8")
    print("\n", out.to_string(index=False))


if __name__ == "__main__":
    main()
