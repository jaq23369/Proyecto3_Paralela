"""Arquitecturas del Lab 3 portadas de Keras/TensorFlow a PyTorch.

Mismas capas, mismo número de parámetros y misma inicialización (Glorot
uniforme, sesgos en cero, BatchNorm con eps=1e-3 y momentum equivalente al de
Keras) que ``secuencial/lab3_original/models.py``. El modelo de Random
Forest se mantiene en scikit-learn y SOLO usa CPU.

Las redes se construyen en CPU y se mueven al dispositivo con ``.to(device)``;
así el mismo código sirve para el baseline en CPU y para la GPU con CUDA.
"""

from __future__ import annotations

import torch
from torch import nn

from comun.config import NUM_CLASSES


def _init_keras_like(module: nn.Module) -> None:
    """Glorot uniforme + bias 0, como los valores por defecto de Keras."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def _bn(channels: int, two_d: bool = True) -> nn.Module:
    cls = nn.BatchNorm2d if two_d else nn.BatchNorm1d
    # Keras: momentum=0.99 (sobre el promedio móvil) == 0.01 en PyTorch; epsilon=1e-3.
    return cls(channels, eps=1e-3, momentum=0.01)


def build_cnn_baseline(num_classes: int = NUM_CLASSES) -> nn.Module:
    """CNN #1: dos bloques conv-pool y una capa densa (2,120,413 parámetros)."""
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
        nn.Linear(128, num_classes),  # logits; softmax va dentro de CrossEntropyLoss
    )
    model.apply(_init_keras_like)
    return model


def build_cnn_regularized(num_classes: int = NUM_CLASSES) -> nn.Module:
    """CNN #2: tres bloques conv + BatchNorm + Dropout (2,246,589 parámetros)."""
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), _bn(32),
        nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), _bn(32),
        nn.MaxPool2d(2), nn.Dropout(0.25),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), _bn(64),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), _bn(64),
        nn.MaxPool2d(2), nn.Dropout(0.25),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), _bn(128),
        nn.MaxPool2d(2), nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256), nn.ReLU(), _bn(256, two_d=False), nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    model.apply(_init_keras_like)
    return model


def build_mlp(
    num_classes: int = NUM_CLASSES,
    hidden_units: tuple[int, ...] = (512, 256),
    dropout_rate: float = 0.30,
    in_features: int = 64 * 64 * 3,
) -> nn.Module:
    """Red fully-connected (6,430,749 parámetros con 512-256)."""
    layers: list[nn.Module] = [nn.Flatten()]
    previous = in_features
    for units in hidden_units:
        layers += [nn.Linear(previous, units), nn.ReLU(), nn.Dropout(dropout_rate)]
        previous = units
    layers.append(nn.Linear(previous, num_classes))
    model = nn.Sequential(*layers)
    model.apply(_init_keras_like)
    return model


BUILDERS = {"cnn1": build_cnn_baseline, "cnn2": build_cnn_regularized, "mlp": build_mlp}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in BUILDERS:
        raise KeyError(f"Modelo desconocido: {name}. Opciones: {sorted(BUILDERS)}")
    return BUILDERS[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Total al estilo Keras: parámetros + media/varianza móviles de BatchNorm."""
    params = sum(p.numel() for p in model.parameters())
    stats = sum(b.numel() for n, b in model.named_buffers() if n.endswith(("running_mean", "running_var")))
    return params + stats


def build_random_forest(n_estimators: int = 400, max_depth: int | None = None, random_state: int = 42):
    """Random Forest del Lab 3 (scikit-learn). Permanece en CPU: no usa CUDA."""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
    )
