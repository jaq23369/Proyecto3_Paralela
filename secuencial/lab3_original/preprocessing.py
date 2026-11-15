"""Utilidades reproducibles para dividir y preprocesar imágenes ASL.

Los datos crudos nunca se modifican. El split se representa con un manifiesto
CSV y cada imagen se redimensiona/normaliza únicamente al cargarla.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

DEFAULT_IMAGE_SIZE = (64, 64)
DEFAULT_RANDOM_SEED = 42
REQUIRED_MANIFEST_COLUMNS = {"path", "label"}


def validate_manifest(manifest: pd.DataFrame) -> None:
    """Valida el contrato mínimo del manifiesto antes de dividirlo."""
    missing_columns = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing_columns:
        raise ValueError(
            "Faltan columnas requeridas en el manifiesto: "
            + ", ".join(sorted(missing_columns))
        )
    if manifest.empty:
        raise ValueError("El manifiesto está vacío")
    if manifest[["path", "label"]].isna().any().any():
        raise ValueError("El manifiesto contiene rutas o etiquetas vacías")
    if manifest["path"].duplicated().any():
        raise ValueError("El manifiesto contiene rutas duplicadas")


def create_stratified_splits(
    manifest: pd.DataFrame,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Asigna train/validation/test conservando la proporción de cada clase.

    Devuelve las columnas ``path``, ``label`` y ``split``. Se hacen dos
    divisiones estratificadas: primero train frente al bloque restante y luego
    validation frente a test dentro de ese bloque.
    """
    validate_manifest(manifest)

    total = train_size + validation_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("Las proporciones train/validation/test deben sumar 1")
    if min(train_size, validation_size, test_size) <= 0:
        raise ValueError("Todas las proporciones deben ser mayores que cero")

    base = manifest.loc[:, ["path", "label"]].copy()
    train, remainder = train_test_split(
        base,
        train_size=train_size,
        random_state=seed,
        stratify=base["label"],
    )

    validation_fraction = validation_size / (validation_size + test_size)
    validation, test = train_test_split(
        remainder,
        train_size=validation_fraction,
        random_state=seed,
        stratify=remainder["label"],
    )

    parts = []
    for name, frame in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        part = frame.copy()
        part["split"] = name
        parts.append(part)

    result = pd.concat(parts, ignore_index=True)
    return result.sort_values(["split", "label", "path"]).reset_index(drop=True)


def missing_files(manifest: pd.DataFrame, image_root: Path) -> list[str]:
    """Retorna las rutas del manifiesto que no existen bajo ``image_root``."""
    validate_manifest(manifest)
    root = Path(image_root)
    return [path for path in manifest["path"] if not (root / path).is_file()]


def preprocess_image(
    image_path: Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """Carga una imagen como RGB, la reduce y normaliza sus píxeles a [0, 1]."""
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Las dimensiones de salida deben ser positivas")

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        resized = rgb.resize(image_size, Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32) / 255.0
    return array


def load_preprocessed_batch(
    manifest: pd.DataFrame,
    image_root: Path,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Carga un subconjunto pequeño como arreglos NumPy para análisis o prueba."""
    validate_manifest(manifest)
    root = Path(image_root)
    images = [preprocess_image(root / row.path, image_size) for row in manifest.itertuples()]
    return np.stack(images), manifest["label"].to_numpy()


def iter_preprocessed_batches(
    manifest: pd.DataFrame,
    image_root: Path,
    batch_size: int = 32,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Produce lotes sin cargar las 17,400 imágenes simultáneamente en RAM."""
    if batch_size <= 0:
        raise ValueError("batch_size debe ser mayor que cero")
    validate_manifest(manifest)

    for start in range(0, len(manifest), batch_size):
        batch = manifest.iloc[start : start + batch_size]
        yield load_preprocessed_batch(batch, image_root, image_size)
