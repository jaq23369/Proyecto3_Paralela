"""Entrenamiento y evaluación (fase de cómputo de la red neuronal).

Diseño pensado para CUDA:

* El dataset de ``train`` (12,180 x 3 x 64 x 64 float32 ~ 600 MB) cabe completo
  en los 6 GB de la GPU, así que se transfiere UNA sola vez y cada lote se
  extrae con indexación en GPU. No hay DataLoader ni copias CPU->GPU por
  iteración: el cuello de botella de E/S queda fuera del bucle de entrenamiento.
* No se llama a ``.item()`` dentro del bucle de lotes (fuerza una sincronización
  GPU->CPU por iteración); las métricas se acumulan en tensores del dispositivo
  y se leen una vez por época.
* ``sync(device)`` se llama antes de leer el reloj, porque los kernels CUDA son
  asíncronos y sin sincronizar se mediría solo el tiempo de *lanzamiento*.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from comun.config import BATCH_SIZE, RANDOM_SEED


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible: revise driver / instalación de PyTorch con CUDA")
        return torch.device("cuda")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Transferencia CPU -> dispositivo (cuando device es CUDA, es la copia H2D)."""
    return torch.from_numpy(array).to(device, non_blocking=False)


@dataclass
class FitResult:
    epochs_run: int = 0
    best_epoch: int = 0
    best_val_acc: float = 0.0
    final_train_acc: float = 0.0
    epoch_seconds: list[float] = field(default_factory=list)
    total_seconds: float = 0.0
    history: list[dict] = field(default_factory=list)


@torch.no_grad()
def evaluate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 512
) -> tuple[float, float]:
    """Devuelve (loss, accuracy). ``x`` e ``y`` ya deben estar en el dispositivo del modelo."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = torch.zeros((), device=x.device)
    correct = torch.zeros((), device=x.device)
    for start in range(0, x.shape[0], batch_size):
        xb, yb = x[start : start + batch_size], y[start : start + batch_size]
        logits = model(xb)
        total_loss += criterion(logits, yb)
        correct += (logits.argmax(1) == yb).sum()
    n = x.shape[0]
    return total_loss.item() / n, correct.item() / n


def fit(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    lr: float,
    max_epochs: int,
    batch_size: int = BATCH_SIZE,
    patience: int | None = None,
    seed: int = RANDOM_SEED,
    amp: bool = False,
    verbose: bool = True,
    adam: str = "torch",
) -> FitResult:
    """Entrena con Adam + entropía cruzada.

    ``patience=None`` ejecuta exactamente ``max_epochs`` (útil para medir tiempos con
    la misma carga de trabajo en CPU y GPU). Con ``patience=k`` aplica parada temprana
    sobre el accuracy de validación y restaura los mejores pesos (como el Lab 3).
    """
    device = x_train.device
    model.to(device)
    if adam == "numba":
        # Paso de Adam con el kernel CUDA propio (un hilo por parámetro). Variante opcional: ver kernel_adam_numba.py.
        if amp:
            raise ValueError("--adam numba no es compatible con --amp (GradScaler necesita torch.optim)")
        from paralelo_cuda.kernel_adam_numba import AdamNumba

        optimizer = AdamNumba(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    result = FitResult()
    best_state, bad_epochs = None, 0
    n = x_train.shape[0]
    sync(device)
    t_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        t_epoch = time.perf_counter()
        model.train()
        order = torch.randperm(n, device=device, generator=generator)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)

        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb, yb = x_train[idx], y_train[idx]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach() * xb.shape[0]
            correct += (logits.detach().argmax(1) == yb).sum()

        val_loss, val_acc = evaluate(model, x_val, y_val)  # .item() -> sincroniza
        sync(device)
        epoch_time = time.perf_counter() - t_epoch
        train_acc = correct.item() / n
        record = {
            "epoch": epoch, "train_loss": loss_sum.item() / n, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "seconds": epoch_time,
        }
        result.history.append(record)
        result.epoch_seconds.append(epoch_time)
        result.epochs_run, result.final_train_acc = epoch, train_acc
        if verbose:
            print(f"    época {epoch:>2}  loss={record['train_loss']:.4f}  acc={train_acc:.4f}  "
                  f"val_acc={val_acc:.4f}  ({epoch_time:.1f}s)", flush=True)

        if val_acc > result.best_val_acc:
            result.best_val_acc, result.best_epoch, bad_epochs = val_acc, epoch, 0
            if patience is not None:
                best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if patience is not None and bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    sync(device)
    result.total_seconds = time.perf_counter() - t_start
    return result
