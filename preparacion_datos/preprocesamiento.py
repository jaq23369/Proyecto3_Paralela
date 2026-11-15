"""Preparación de datos: versión SECUENCIAL (original del Lab 3) y PARALELA (CPU multinúcleo).

Separación del pipeline (requisito del proyecto):

* Este módulo = fase de **preparación de datos, estrictamente en CPU**:
  abrir JPG -> convertir a RGB -> reducir 200x200 a 64x64 (Lanczos) -> normalizar.
* ``models.py`` / ``training.py`` = fase de **entrenamiento, en GPU con CUDA**.

El trabajo por imagen es independiente (no hay dependencias entre imágenes),
por lo que el bucle secuencial se reparte entre procesos con
``ProcessPoolExecutor``. Se usan procesos y no hilos porque la decodificación
JPEG/Lanczos de Pillow compite por el GIL en parte de su código y el pipeline
mezcla E/S de disco con cómputo.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from comun.config import IMAGE_SIZE

REQUIRED_MANIFEST_COLUMNS = {"path", "label"}


# ---------------------------------------------------------------------------
# Utilidades de manifiesto
# ---------------------------------------------------------------------------
def validate_manifest(manifest: pd.DataFrame) -> None:
    """Valida el contrato mínimo del manifiesto (código del Lab 3)."""
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError("Faltan columnas en el manifiesto: " + ", ".join(sorted(missing)))
    if manifest.empty:
        raise ValueError("El manifiesto está vacío")
    if manifest[["path", "label"]].isna().any().any():
        raise ValueError("El manifiesto contiene rutas o etiquetas vacías")
    if manifest["path"].duplicated().any():
        raise ValueError("El manifiesto contiene rutas duplicadas")


def missing_files(manifest: pd.DataFrame, image_root: Path) -> list[str]:
    """Rutas del manifiesto que no existen bajo ``image_root``."""
    validate_manifest(manifest)
    root = Path(image_root)
    return [p for p in manifest["path"] if not (root / p).is_file()]


def class_names(manifest: pd.DataFrame) -> list[str]:
    """Clases en orden alfabético estable (define el índice numérico de cada letra)."""
    return sorted(manifest["label"].unique())


def encode_labels(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index[x] for x in labels], dtype=np.int64)


# ---------------------------------------------------------------------------
# VERSIÓN SECUENCIAL (original del Lab 3, sin modificar la lógica)
# ---------------------------------------------------------------------------
def preprocess_image(image_path: Path, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Carga una imagen como RGB, la reduce con Lanczos y normaliza a [0, 1]."""
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Las dimensiones de salida deben ser positivas")
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        resized = rgb.resize(image_size, Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32) / 255.0
    return array


def load_preprocessed_batch(
    manifest: pd.DataFrame, image_root: Path, image_size: tuple[int, int] = IMAGE_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """BASELINE SECUENCIAL: un solo hilo, imagen por imagen (bucle del Lab 3)."""
    validate_manifest(manifest)
    root = Path(image_root)
    images = [preprocess_image(root / row.path, image_size) for row in manifest.itertuples()]
    return np.stack(images), manifest["label"].to_numpy()


# ---------------------------------------------------------------------------
# VERSIÓN PARALELA (CPU multinúcleo)
# ---------------------------------------------------------------------------
def _decode_resize_chunk(args: tuple[str, list[str], tuple[int, int]]) -> np.ndarray:
    """Tarea de un worker: decodifica y reduce un bloque de imágenes -> uint8.

    Devuelve uint8 (no float32) para que el envío entre procesos pese 4x menos;
    la normalización se hace una sola vez, vectorizada, en el proceso padre.
    """
    root, rel_paths, size = args
    out = np.empty((len(rel_paths), size[1], size[0], 3), dtype=np.uint8)
    for i, rel in enumerate(rel_paths):
        with Image.open(os.path.join(root, rel)) as image:
            out[i] = np.asarray(
                image.convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.uint8
            )
    return out


def default_workers() -> int:
    return os.cpu_count() or 1


def load_preprocessed_parallel(
    manifest: pd.DataFrame,
    image_root: Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
    workers: int | None = None,
    chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """VERSIÓN PARALELA: reparte bloques de ``chunk_size`` imágenes entre ``workers`` procesos.

    Produce exactamente el mismo resultado que ``load_preprocessed_batch``
    (verificado en ``tests/test_preprocessing.py``). El orden se conserva porque
    ``Executor.map`` devuelve los resultados en el orden de envío.
    El tiempo medido incluye la creación del pool de procesos.
    """
    validate_manifest(manifest)
    workers = workers or default_workers()
    paths = manifest["path"].tolist()
    root = str(image_root)
    tasks = [(root, paths[i : i + chunk_size], tuple(image_size)) for i in range(0, len(paths), chunk_size)]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        chunks = list(pool.map(_decode_resize_chunk, tasks))

    images_u8 = np.concatenate(chunks, axis=0)
    images = images_u8.astype(np.float32) / 255.0
    return images, manifest["label"].to_numpy()


def to_nchw(images_nhwc: np.ndarray) -> np.ndarray:
    """(N, H, W, C) -> (N, C, H, W), el formato que espera PyTorch (contiguo)."""
    return np.ascontiguousarray(images_nhwc.transpose(0, 3, 1, 2))
