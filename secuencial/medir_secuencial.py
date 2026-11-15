"""Mide la solución ORIGINAL del Lab 3 (Keras/TensorFlow en CPU) — línea base del Paso 2.

Las instrucciones del proyecto piden "ejecutar la solución secuencial desarrollada previamente y medir el
tiempo". Este script lo hace con el código del laboratorio SIN modificar (``secuencial/lab3_original/``):

  * preprocesamiento : ``preprocessing.load_preprocessed_batch`` (bucle secuencial del Lab 3)
  * modelos          : ``models.build_cnn_baseline / build_cnn_regularized / build_mlp`` (Keras)
  * entrenamiento    : ``model.fit(batch_size=64, ...)`` con los learning rates ganadores del Lab 3
  * inferencia       : ``model.evaluate`` sobre test

Mide exactamente las mismas fases, con las mismas imágenes, semilla, lote y épocas fijas que
``paralelo_cuda/medir_cuda.py``, y escribe el mismo formato de CSV (modo ``keras_original``) para que
``comparacion/comparar.py`` lo incluya. TensorFlow en Windows nativo corre solo en CPU (sin soporte de GPU desde
TF 2.11); se deja la configuración de hilos POR DEFECTO de TensorFlow, como corrió el laboratorio.

Se ejecuta con un entorno que tenga TensorFlow (no el de PyTorch):

    set ASL_DATA_ROOT=C:\\datasets\\asl
    python secuencial/medir_secuencial.py --tag pc1 --reps 3 --epochs 3
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                                   # paquete comun/ (rutas, monitor)
sys.path.insert(0, str(ROOT / "secuencial" / "lab3_original"))  # preprocessing.py / models.py ORIGINALES del Lab 3

from comun import config  # noqa: E402  (solo usa os/pathlib: funciona en el entorno de TensorFlow)
from comun.monitor import ResourceMonitor  # noqa: E402

TRAIN_DIR, SPLIT_MANIFEST = config.TRAIN_DIR, config.SPLIT_MANIFEST
RESULTS_DIR, LOGS_DIR = config.RESULTS_DIR, config.LOGS_DIR
SEED, BATCH_SIZE = config.RANDOM_SEED, config.BATCH_SIZE
LR = {name: spec["lr"] for name, spec in config.MODEL_SPECS.items()}   # 1e-3, 3e-4, 1e-4 (Lab 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=["cnn1", "cnn2", "mlp"], choices=["cnn1", "cnn2", "mlp"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-monitor", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import tensorflow as tf

    import models as lab_models            # secuencial/lab3_original/models.py
    import preprocessing as lab_prep       # secuencial/lab3_original/preprocessing.py

    gpus = tf.config.list_physical_devices("GPU")
    print(f"== Keras original (Lab 3) | TensorFlow {tf.__version__} | GPUs visibles para TF: {len(gpus)} "
          f"| hilos intra-op: {tf.config.threading.get_intra_op_parallelism_threads() or 'auto'}")

    manifest = pd.read_csv(SPLIT_MANIFEST)
    if args.limit:
        manifest = manifest.sample(n=min(args.limit, len(manifest)), random_state=SEED)
    manifest = manifest.reset_index(drop=True)
    classes = sorted(manifest["label"].unique())
    index = {c: i for i, c in enumerate(classes)}

    builders = {
        "cnn1": lab_models.build_cnn_baseline,
        "cnn2": lab_models.build_cnn_regularized,
        "mlp": lambda: lab_models.build_mlp(hidden_units=(512, 256), dropout_rate=0.30),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    rows = []
    monitor = None if args.no_monitor else ResourceMonitor(LOGS_DIR / f"resources_{args.tag}_keras_original.csv")

    def record(rep, phase, model, seconds, **extra):
        rows.append({"tag": args.tag, "mode": "keras_original", "rep": rep, "phase": phase,
                     "model": model, "seconds": round(seconds, 4), **extra})
        print(f"   [rep {rep}] {phase:<12} {model:<5} {seconds:8.2f} s  {extra if extra else ''}", flush=True)

    def mark(label):
        if monitor:
            monitor.mark(label)

    try:
        if monitor:
            monitor.__enter__()
        for rep in range(1, args.reps + 1):
            print(f"\n--- Repetición {rep}/{args.reps} ---")
            mark(f"preprocess_r{rep}")
            t0 = time.perf_counter()
            images, labels = lab_prep.load_preprocessed_batch(manifest, TRAIN_DIR)   # código original, NHWC
            record(rep, "preprocess", "-", time.perf_counter() - t0, images=len(manifest))
            y_all = np.array([index[x] for x in labels], dtype=np.int64)
            split = manifest["split"].to_numpy()
            x_tr, y_tr = images[split == "train"], y_all[split == "train"]
            x_va, y_va = images[split == "validation"], y_all[split == "validation"]
            x_te, y_te = images[split == "test"], y_all[split == "test"]
            del images  # libera ~0.85 GB: evita tener el dataset duplicado en RAM junto a TensorFlow

            for name in args.models:
                tf.keras.utils.set_random_seed(SEED)
                tf.keras.backend.clear_session()
                model = builders[name]()
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR[name]),
                              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                mark(f"train_{name}_r{rep}")
                t0 = time.perf_counter()
                hist = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=args.epochs,
                                 batch_size=BATCH_SIZE, verbose=0)
                dt = time.perf_counter() - t0
                record(rep, "train", name, dt, epochs=args.epochs, sec_per_epoch=round(dt / args.epochs, 3),
                       val_acc=round(float(max(hist.history["val_accuracy"])), 4), params=int(model.count_params()))
                mark(f"infer_{name}_r{rep}")
                t0 = time.perf_counter()
                _, test_acc = model.evaluate(x_te, y_te, batch_size=512, verbose=0)
                record(rep, "infer", name, time.perf_counter() - t0, test_acc=round(float(test_acc), 4),
                       images=int(len(x_te)))
            mark("idle")
    finally:
        if monitor:
            monitor.__exit__(None, None, None)
        if rows:
            out = RESULTS_DIR / f"benchmark_{args.tag}_keras_original.csv"
            keys = sorted({k for r in rows for k in r},
                          key=lambda k: (k not in ("tag", "mode", "rep", "phase", "model", "seconds"), k))
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
            (RESULTS_DIR / f"keras_env_{args.tag}.json").write_text(json.dumps({
                "tensorflow": tf.__version__, "keras": tf.keras.__version__,
                "gpus_visible_to_tensorflow": len(gpus),
                "nota": "TensorFlow en Windows nativo no usa GPU (sin soporte desde TF 2.11)."}, indent=2),
                encoding="utf-8")
            print(f"\nResultados guardados en {out}")


if __name__ == "__main__":
    main()
