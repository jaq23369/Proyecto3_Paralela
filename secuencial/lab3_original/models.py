"""Arquitecturas candidatas para el clasificador de letras ASL.

Cada función construye y compila un modelo sin entrenarlo (ninguna llama a
`.fit`). Quedan listas para recibir los lotes que produce
`preprocessing.py` en la entrega final.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from tensorflow.keras import layers

DEFAULT_INPUT_SHAPE = (64, 64, 3)
NUM_CLASSES = 29


def build_cnn_baseline(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """CNN simple: dos bloques conv-pool y una capa densa.

    Sirve como punto de referencia rápido antes de evaluar arquitecturas
    más profundas o regularizadas.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn_baseline",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_regularized(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
) -> keras.Model:
    """CNN más profunda con BatchNorm y Dropout para controlar overfitting.

    Tres bloques conv-conv-pool con normalización por lotes, seguidos de
    una capa densa con dropout fuerte antes de la salida.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn_regularizada",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mlp(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    hidden_units: tuple[int, ...] = (512, 256),
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Red fully-connected: aplana la imagen y descarta la estructura 2D.

    Sirve de comparación contra las CNN para mostrar cuánto aporta la
    convolución/pooling frente a una red densa equivalente. ``hidden_units``
    permite comparar profundidades y anchos distintos sin duplicar la
    definición del modelo.
    """
    hidden_layers = []
    for units in hidden_units:
        hidden_layers.extend(
            [
                layers.Dense(units, activation="relu"),
                layers.Dropout(dropout_rate),
            ]
        )

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Flatten(),
            *hidden_layers,
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="mlp",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_random_forest(
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Algoritmo alterno no basado en Deep Learning: Random Forest.

    Se entrena sobre los píxeles aplanados de las imágenes preprocesadas.
    Sirve de referencia clásica para contrastar contra las redes
    neuronales: no requiere GPU, es rápido de ajustar y da una primera
    cota de desempeño interpretable (importancia de variables) antes de
    invertir tiempo de cómputo en las CNN.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
