"""Pruebas de corrección: lo paralelo debe dar el mismo resultado que lo secuencial."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comun import config  # noqa: E402
from paralelo_cuda.modelos import build_model, count_parameters  # noqa: E402
from preparacion_datos.preprocesamiento import (  # noqa: E402
    load_preprocessed_batch, load_preprocessed_parallel, to_nchw,
)

requires_data = pytest.mark.skipif(
    not config.TRAIN_DIR.is_dir(), reason="dataset no encontrado (defina ASL_DATA_ROOT)"
)


@requires_data
def test_parallel_matches_sequential_exactly():
    manifest = pd.read_csv(config.SPLIT_MANIFEST).head(200)
    x_seq, y_seq = load_preprocessed_batch(manifest, config.TRAIN_DIR)
    x_par, y_par = load_preprocessed_parallel(manifest, config.TRAIN_DIR, workers=4, chunk_size=16)
    assert x_seq.shape == x_par.shape == (200, 64, 64, 3)
    assert x_seq.dtype == x_par.dtype == np.float32
    np.testing.assert_array_equal(x_seq, x_par)
    np.testing.assert_array_equal(y_seq, y_par)
    assert 0.0 <= x_par.min() and x_par.max() <= 1.0


def test_to_nchw_layout():
    x = np.zeros((5, 64, 64, 3), dtype=np.float32)
    assert to_nchw(x).shape == (5, 3, 64, 64)


# Número de parámetros que reporta Keras para estos modelos en el Lab 3
@pytest.mark.parametrize(
    "name,expected", [("cnn1", 2_120_413), ("cnn2", 2_246_589), ("mlp", 6_430_749)]
)
def test_param_counts_match_lab3(name, expected):
    spec = config.MODEL_SPECS[name]
    assert count_parameters(build_model(name, **spec["kwargs"])) == expected


@pytest.mark.parametrize("name", ["cnn1", "cnn2", "mlp"])
def test_forward_shape(name):
    model = build_model(name, **config.MODEL_SPECS[name]["kwargs"]).eval()
    out = model(torch.zeros(2, 3, 64, 64))
    assert out.shape == (2, config.NUM_CLASSES)
