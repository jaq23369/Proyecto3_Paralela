from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class DatasetInfo:
    class_names: list[str]
    counts_per_class: pd.Series


def list_class_dirs(train_dir: Path) -> list[Path]:
    return sorted(p for p in Path(train_dir).iterdir() if p.is_dir())


def count_images_per_class(train_dir: Path) -> pd.Series:
    counts = {
        class_dir.name: sum(
            1 for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        for class_dir in list_class_dirs(train_dir)
    }
    return pd.Series(counts, name="n_images").sort_index()


def describe_dataset(train_dir: Path) -> DatasetInfo:
    counts = count_images_per_class(train_dir)
    return DatasetInfo(class_names=list(counts.index), counts_per_class=counts)


def stratified_sample(
    train_dir: Path, n_per_class: int, seed: int = 42
) -> pd.DataFrame:
    """Muestra estratificada: hasta n_per_class imágenes por clase.

    Retorna un DataFrame con columnas [path, label], con path relativo a train_dir.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    for class_dir in list_class_dirs(train_dir):
        files = sorted(
            f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        chosen = rng.sample(files, min(n_per_class, len(files)))
        rows.extend(
            {"path": f.relative_to(train_dir).as_posix(), "label": class_dir.name}
            for f in chosen
        )
    return pd.DataFrame(rows)


def load_image(train_dir: Path, relative_path: str) -> Image.Image:
    return Image.open(Path(train_dir) / relative_path)


def image_shapes_sample(train_dir: Path, n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Revisa resolución/formato de una muestra aleatoria de imágenes."""
    rng = random.Random(seed)
    all_files = [
        f
        for class_dir in list_class_dirs(train_dir)
        for f in class_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    sampled = rng.sample(all_files, min(n, len(all_files)))
    rows = []
    for f in sampled:
        with Image.open(f) as img:
            rows.append(
                {
                    "file": f.name,
                    "class": f.parent.name,
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format,
                }
            )
    return pd.DataFrame(rows)
