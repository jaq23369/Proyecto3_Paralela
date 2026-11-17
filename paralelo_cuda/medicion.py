"""Benchmark de las fases del pipeline ASL: secuencial vs. CPU multihilo vs. CUDA.

Mide, con las MISMAS imágenes, modelos, semillas, lotes y número de épocas:

  preprocess        : disco -> tensor normalizado (17,400 imágenes) .... CPU
  h2d_transfer      : copia CPU -> GPU de train/val/test ............... solo CUDA
  train_<modelo>    : entrenamiento de ``--epochs`` épocas fijas
  infer_<modelo>    : predicción sobre el conjunto test (2,610 imágenes)

Configuraciones (``--mode``):

  sequential : preprocesamiento en 1 proceso + PyTorch en CPU con 1 hilo   (BASELINE)
  cpu_mt     : preprocesamiento multiproceso + PyTorch en CPU, todos los hilos
  cuda       : preprocesamiento multiproceso (CPU) + entrenamiento en GPU (CUDA)

Uso::

    python paralelo_cuda/medir_cuda.py --mode sequential --tag pc1 --reps 3 --epochs 3
    python paralelo_cuda/medir_cuda.py --mode cuda       --tag pc1 --reps 3 --epochs 3

Genera ``results/benchmark_<tag>_<mode>.csv``, ``results/machine_<tag>.json`` y el
log de recursos ``logs/resources_<tag>_<mode>.csv`` (CPU por núcleo + GPU).
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

from comun import config  # noqa: E402
from paralelo_cuda.modelos import build_model, count_parameters  # noqa: E402
from comun.monitor import ResourceMonitor  # noqa: E402
from preparacion_datos.preprocesamiento import (  # noqa: E402
    class_names, encode_labels, load_preprocessed_batch, load_preprocessed_parallel, to_nchw,
)
from paralelo_cuda.entrenamiento import evaluate, fit, get_device, set_seed, sync, to_device  # noqa: E402

from comun.config import MODES  # noqa: E402


def warm_file_cache(paths: list[Path]) -> None:
    """Lee todos los archivos una vez para que ninguna configuración pague E/S en frío."""
    def _read(p: Path) -> int:
        with open(p, "rb") as f:
            return len(f.read())
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_read, paths))


def cpu_name() -> str:
    """Nombre comercial del CPU (platform.processor() en Windows solo da la familia)."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or "desconocido"


def machine_info(device_name: str) -> dict:
    info = {
        "cpu": cpu_name(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "os": platform.platform(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu=props.name, gpu_vram_gb=round(props.total_memory / 2**30, 1),
            gpu_sm_count=props.multi_processor_count,
            gpu_compute_capability=f"{props.major}.{props.minor}",
        )
    return info


def run(args) -> None:
    """Ejecuta el benchmark con los argumentos ya parseados (ver paralelo_cuda/medir_cuda.py)."""
    cfg = MODES[args.mode]
    torch.set_num_threads(cfg["threads"])
    device = get_device(cfg["device"])
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    manifest = pd.read_csv(config.SPLIT_MANIFEST)
    if args.limit:
        manifest = manifest.sample(n=min(args.limit, len(manifest)), random_state=config.RANDOM_SEED)
    manifest = manifest.reset_index(drop=True)
    classes = class_names(manifest)
    n_images = len(manifest)

    results_dir, logs_dir = config.RESULTS_DIR, config.LOGS_DIR
    results_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    out_csv = results_dir / f"benchmark_{args.tag}_{args.mode}.csv"
    (results_dir / f"machine_{args.tag}.json").write_text(
        json.dumps(machine_info(device.type), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"== Benchmark | tag={args.tag} mode={args.mode} device={device} "
          f"threads={cfg['threads']} reps={args.reps} epochs={args.epochs} imágenes={n_images}")
    print("   calentando caché de archivos ...", flush=True)
    warm_file_cache([config.TRAIN_DIR / p for p in manifest["path"]])

    rows: list[dict] = []
    monitor = ResourceMonitor(logs_dir / f"resources_{args.tag}_{args.mode}.csv") if not args.no_monitor else None

    def record(rep: int, phase: str, model: str, seconds: float, **extra) -> None:
        rows.append({"tag": args.tag, "mode": args.mode, "rep": rep, "phase": phase,
                     "model": model, "seconds": round(seconds, 4), **extra})
        print(f"   [rep {rep}] {phase:<16} {model:<5} {seconds:8.2f} s  {extra if extra else ''}", flush=True)

    def mark(label: str) -> None:
        if monitor:
            monitor.mark(label)

    try:
        if monitor:
            monitor.__enter__()

        # --- Calentamiento CUDA (inicialización de contexto + autotune cuDNN), reportado aparte ---
        if device.type == "cuda":
            mark("cuda_warmup")
            t0 = time.perf_counter()
            for name in args.models:
                set_seed()
                m = build_model(name, **config.MODEL_SPECS[name]["kwargs"]).to(device)
                xw = torch.rand(256, 3, 64, 64, device=device)
                yw = torch.randint(0, config.NUM_CLASSES, (256,), device=device)
                fit(m, xw, yw, xw[:64], yw[:64], lr=1e-3, max_epochs=1, amp=args.amp, verbose=False, adam=args.adam)
            sync(device)
            record(0, "cuda_warmup", "all", time.perf_counter() - t0)

        for rep in range(1, args.reps + 1):
            print(f"\n--- Repetición {rep}/{args.reps} ---")

            # ============ FASE 1: preprocesamiento (CPU) ============
            mark(f"preprocess_r{rep}")
            t0 = time.perf_counter()
            if cfg["parallel_prep"]:
                images, labels = load_preprocessed_parallel(manifest, config.TRAIN_DIR, config.IMAGE_SIZE)
            else:
                images, labels = load_preprocessed_batch(manifest, config.TRAIN_DIR, config.IMAGE_SIZE)
            record(rep, "preprocess", "-", time.perf_counter() - t0, images=n_images)

            y_all = encode_labels(labels, classes)
            x_all = to_nchw(images)
            del images
            split = manifest["split"].to_numpy()
            parts = {s: (x_all[split == s], y_all[split == s]) for s in ("train", "validation", "test")}

            # ============ FASE 2: transferencia al dispositivo ============
            mark(f"h2d_r{rep}")
            sync(device)
            t0 = time.perf_counter()
            tensors = {s: (to_device(x, device), to_device(y, device)) for s, (x, y) in parts.items()}
            sync(device)
            if device.type == "cuda":
                mb = sum(t.numel() * t.element_size() for pair in tensors.values() for t in pair) / 2**20
                record(rep, "h2d_transfer", "-", time.perf_counter() - t0, megabytes=round(mb))
            x_tr, y_tr = tensors["train"]
            x_va, y_va = tensors["validation"]
            x_te, y_te = tensors["test"]

            # ============ FASE 3 y 4: entrenamiento e inferencia ============
            for name in args.models:
                spec = config.MODEL_SPECS[name]
                set_seed()
                model = build_model(name, **spec["kwargs"]).to(device)

                mark(f"train_{name}_r{rep}")
                result = fit(model, x_tr, y_tr, x_va, y_va, lr=spec["lr"], max_epochs=args.epochs,
                             patience=None, amp=args.amp, verbose=False, adam=args.adam)
                record(rep, "train", name, result.total_seconds, epochs=result.epochs_run,
                       sec_per_epoch=round(float(np.mean(result.epoch_seconds)), 3),
                       val_acc=round(result.best_val_acc, 4), params=count_parameters(model))

                mark(f"infer_{name}_r{rep}")
                sync(device)
                t0 = time.perf_counter()
                _, test_acc = evaluate(model, x_te, y_te)
                sync(device)
                record(rep, "infer", name, time.perf_counter() - t0, test_acc=round(test_acc, 4),
                       images=int(x_te.shape[0]))

            mark("idle")
            if device.type == "cuda":
                del tensors, x_tr, y_tr, x_va, y_va, x_te, y_te, model
                torch.cuda.empty_cache()
    finally:
        if monitor:
            monitor.__exit__(None, None, None)
        if rows:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in
                              ("tag", "mode", "rep", "phase", "model", "seconds"), k))
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nResultados guardados en {out_csv}")

