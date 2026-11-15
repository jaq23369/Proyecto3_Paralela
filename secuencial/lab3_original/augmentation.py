"""Data augmentation segura para señas ASL.

Solo incluye transformaciones que preservan el significado de una seña.
Deliberadamente no incluye `RandomFlip`: voltear una mano horizontalmente
puede convertir una letra en otra o en una seña que no existe en ASL, así
que el flip no es una transformación segura por defecto para este dataset.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

DEFAULT_ROTATION_FACTOR = 15 / 360  # ±15°
DEFAULT_TRANSLATION_FACTOR = 0.1
DEFAULT_ZOOM_FACTOR = 0.1
DEFAULT_BRIGHTNESS_FACTOR = 0.1
DEFAULT_CONTRAST_FACTOR = 0.1


def build_augmentation(
    rotation_factor: float = DEFAULT_ROTATION_FACTOR,
    translation_factor: float = DEFAULT_TRANSLATION_FACTOR,
    zoom_factor: float = DEFAULT_ZOOM_FACTOR,
    brightness_factor: float = DEFAULT_BRIGHTNESS_FACTOR,
    contrast_factor: float = DEFAULT_CONTRAST_FACTOR,
    seed: int = 42,
) -> keras.Sequential:
    """Pipeline de augmentation aplicable solo a `train`, después del split.

    Simula variaciones de cámara y de mano que no cambian la letra: ángulo,
    posición, distancia e iluminación. No reordena ni recorta la mano fuera
    de cuadro de forma agresiva, para no perder la seña completa.
    """
    return keras.Sequential(
        [
            layers.RandomRotation(rotation_factor, seed=seed),
            layers.RandomTranslation(
                translation_factor, translation_factor, seed=seed
            ),
            layers.RandomZoom(zoom_factor, seed=seed),
            layers.RandomBrightness(brightness_factor, value_range=(0, 1), seed=seed),
            layers.RandomContrast(contrast_factor, seed=seed),
        ],
        name="augmentation_segura",
    )
