"""Configuración central del proyecto (rutas, semillas e hiperparámetros del Lab 3).

Los hiperparámetros replican EXACTAMENTE los que ganaron en el Laboratorio 3
de Data Science, para que la comparación secuencial vs. CUDA mida solo el efecto
del hardware y no un cambio de modelo.

La ruta del dataset se lee de la variable de entorno ``ASL_DATA_ROOT``. Esto
permite guardar las ~87,000 imágenes FUERA de OneDrive (que intentaría
sincronizarlas) y que cada integrante use su propia ruta.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Configuraciones del benchmark (ver paralelo_cuda/medir_cuda.py)
MODES = {
    "sequential": {"device": "cpu", "parallel_prep": False, "threads": 1},
    "cpu_mt": {"device": "cpu", "parallel_prep": True, "threads": os.cpu_count()},
    "cuda": {"device": "cuda", "parallel_prep": True, "threads": os.cpu_count()},
}

# Carpeta que contiene asl_alphabet_train/ y asl_alphabet_test/
DATA_ROOT = Path(os.environ.get("ASL_DATA_ROOT", PROJECT_ROOT / "data"))
TRAIN_DIR = DATA_ROOT / "asl_alphabet_train"

SPLIT_MANIFEST = PROJECT_ROOT / "data" / "split_manifest.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"

IMAGE_SIZE = (64, 64)
NUM_CLASSES = 29
RANDOM_SEED = 42

# --- Hiperparámetros de entrenamiento heredados del Lab 3 -------------------
BATCH_SIZE = 64
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 4

# nombre -> (learning rate ganador del Lab 3, kwargs del constructor)
MODEL_SPECS = {
    "cnn1": {"lr": 1e-3, "kwargs": {}},
    "cnn2": {"lr": 3e-4, "kwargs": {}},
    "mlp": {"lr": 1e-4, "kwargs": {"hidden_units": (512, 256), "dropout_rate": 0.30}},
}
