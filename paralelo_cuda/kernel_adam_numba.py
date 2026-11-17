"""Kernel CUDA propio (Numba) para el paso de Adam: «CUDA Python» explícito.

Es la operación 14 del diseño (``docs/04``, tabla de hilos/bloques): **un hilo por parámetro**,
bloques de 256 hilos y grid de ⌈N / 256⌉ bloques (CNN #2: N = 2,245,437 -> 8,772 bloques).
Cada hilo hace la actualización de Adam de SU parámetro, sin comunicarse con los demás:

    m <- b1*m + (1-b1)*g          v <- b2*v + (1-b2)*g²
    p <- p - (lr / (1-b1^t)) * m / ( sqrt(v)/sqrt(1-b2^t) + eps )

(misma fórmula que ``torch.optim.Adam``, para poder validar el resultado con tolerancia numérica).

Los tensores se comparten con PyTorch SIN copiarlos: ``numba.cuda.as_cuda_array`` envuelve la
memoria de la GPU que ya posee el tensor (protocolo ``__cuda_array_interface__``).

Uso (validación + medición del kernel por separado)::

    python paralelo_cuda/kernel_adam_numba.py --tag pc2 --model cnn2
Salida: results/kernel_adam_numba_<tag>.csv
"""

from __future__ import annotations

import math
import warnings

# OJO: nvJitLink de CUDA 12.9 debe cargarse ANTES que torch. Si torch se importa primero, el proceso queda con
# su nvJitLink 12.8 y Numba falla al enlazar el kernel ("may need newer version of nvJitLink").
from cuda.bindings import nvjitlink  # isort: skip

nvjitlink.version()  # fuerza la carga de la DLL ahora
from numba import cuda  # isort: skip

import torch  # isort: skip

# Los tensores pequeños (sesgos) lanzan pocos bloques; es esperado y no afecta al total.
warnings.filterwarnings("ignore", message=".*under-utilization.*")

THREADS_PER_BLOCK = 256  # múltiplo de 32 (warp), igual que la tabla de docs/04


@cuda.jit
def adam_kernel(p, g, m, v, lr_corr, beta1, beta2, inv_sqrt_bc2, eps):
    """Un hilo = un parámetro. ``p, g, m, v`` son vectores 1-D del mismo tamaño."""
    i = cuda.grid(1)
    if i < p.size:
        gi = g[i]
        mi = beta1 * m[i] + (1.0 - beta1) * gi
        vi = beta2 * v[i] + (1.0 - beta2) * gi * gi
        m[i] = mi
        v[i] = vi
        p[i] -= lr_corr * mi / (math.sqrt(vi) * inv_sqrt_bc2 + eps)


class AdamNumba:
    """Optimizador Adam cuyo paso se ejecuta con el kernel de Numba (interfaz mínima de torch.optim)."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.beta1, self.beta2, self.eps = lr, betas[0], betas[1], eps
        self.step_count = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.n_params = sum(p.numel() for p in self.params)
        # Los envoltorios de Numba para p, m y v no cambian entre pasos: se crean una sola vez.
        self._p_nb = [cuda.as_cuda_array(p.detach().view(-1)) for p in self.params]
        self._m_nb = [cuda.as_cuda_array(m.view(-1)) for m in self.m]
        self._v_nb = [cuda.as_cuda_array(v.view(-1)) for v in self.v]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.step_count += 1
        t = self.step_count
        lr_corr = self.lr / (1.0 - self.beta1**t)
        inv_sqrt_bc2 = 1.0 / math.sqrt(1.0 - self.beta2**t)
        for p, p_nb, m_nb, v_nb in zip(self.params, self._p_nb, self._m_nb, self._v_nb):
            if p.grad is None:
                continue
            blocks = (p.numel() + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            adam_kernel[blocks, THREADS_PER_BLOCK](
                p_nb, cuda.as_cuda_array(p.grad.detach().contiguous().view(-1)), m_nb, v_nb,
                lr_corr, self.beta1, self.beta2, inv_sqrt_bc2, self.eps,
            )


def grid_for(n_params: int) -> tuple[int, int]:
    """(bloques, hilos por bloque) para N parámetros en UN solo lanzamiento."""
    return (n_params + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, THREADS_PER_BLOCK


def main() -> None:
    import argparse
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pandas as pd

    from comun import config
    from paralelo_cuda.modelos import build_model

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="cnn2", choices=["cnn1", "cnn2", "mlp"])
    ap.add_argument("--steps", type=int, default=100, help="pasos de Adam medidos")
    args = ap.parse_args()

    device = torch.device("cuda")
    spec = config.MODEL_SPECS[args.model]

    def make():
        torch.manual_seed(config.RANDOM_SEED)
        return build_model(args.model, **spec["kwargs"]).to(device)

    # Dos copias idénticas del modelo y gradientes idénticos en cada paso.
    ref_model, num_model = make(), make()
    ref_opt = torch.optim.Adam(ref_model.parameters(), lr=1e-3)
    num_opt = AdamNumba(num_model.parameters(), lr=1e-3)
    n = num_opt.n_params
    blocks, threads = grid_for(n)
    print(f"{args.model}: {n:,} parámetros | kernel Adam: {threads} hilos/bloque, "
          f"~{blocks:,} bloques si fuera 1 lanzamiento (aquí 1 por tensor: {len(num_opt.params)} lanzamientos)")

    gen = torch.Generator(device=device)
    gen.manual_seed(7)

    def set_grads():
        for pr, pn in zip(ref_model.parameters(), num_model.parameters()):
            g = torch.randn(pr.shape, device=device, generator=gen)
            pr.grad, pn.grad = g.clone(), g.clone()

    # 1) Validación contra torch.optim.Adam
    for _ in range(10):
        set_grads()
        ref_opt.step()
        num_opt.step()
    torch.cuda.synchronize()
    max_err = max((a - b).abs().max().item() for a, b in zip(ref_model.parameters(), num_model.parameters()))
    ok = max_err < 1e-5
    print(f"validación (10 pasos): error máximo |torch - numba| = {max_err:.3e}  ->  {'OK' if ok else 'FALLA'}")

    # 2) Medición del kernel por separado (gradientes fijos, solo el paso del optimizador)
    set_grads()
    rows = []
    for name, opt in (("torch.optim.Adam", ref_opt), ("Numba kernel", num_opt)):
        for _ in range(10):  # calentamiento (compilación JIT de Numba incluida)
            opt.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            opt.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / args.steps * 1e3
        rows.append({"tag": args.tag, "model": args.model, "optimizer": name, "params": n,
                     "threads_per_block": threads, "ms_per_step": round(ms, 4),
                     "max_abs_err_vs_torch": max_err, "valid": ok})
        print(f"  {name:<18} {ms:8.4f} ms/paso")
    out = config.RESULTS_DIR / f"kernel_adam_numba_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
