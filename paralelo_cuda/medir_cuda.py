"""Benchmark de las fases del pipeline ASL: secuencial vs. CPU multihilo vs. CUDA.

Mide, con las MISMAS imágenes, modelos, semillas, lotes y número de épocas:

  preprocess        : disco -> tensor normalizado (17,400 imágenes) .... CPU
  h2d_transfer      : copia CPU -> GPU de train/val/test ............... solo CUDA
  train             : entrenamiento de ``--epochs`` épocas fijas (por modelo)
  infer             : predicción sobre el conjunto test, 2,610 imágenes (por modelo)

Configuraciones (``--mode``):

  sequential : preprocesamiento en 1 proceso + PyTorch en CPU con 1 hilo   (BASELINE)
  cpu_mt     : preprocesamiento multiproceso + PyTorch en CPU, todos los hilos
  cuda       : preprocesamiento multiproceso (CPU) + entrenamiento en GPU (CUDA)

Ejemplos::

    set ASL_DATA_ROOT=C:\\datasets\\asl
    python paralelo_cuda/medir_cuda.py --mode sequential --tag pc1 --reps 3 --epochs 3
    python paralelo_cuda/medir_cuda.py --mode cpu_mt     --tag pc1 --reps 3 --epochs 3
    python paralelo_cuda/medir_cuda.py --mode cuda       --tag pc1 --reps 3 --epochs 3

Genera ``results/benchmark_<tag>_<mode>.csv``, ``results/machine_<tag>.json`` y el
log de recursos ``logs/resources_<tag>_<mode>.csv`` (CPU por núcleo + GPU).

Nota: este archivo es deliberadamente liviano. En Windows, los procesos hijos del
preprocesamiento paralelo reimportan el script principal (``spawn``); si aquí se
importara ``torch`` cada worker pagaría varios segundos de arranque. Por eso la
lógica pesada vive en ``paralelo_cuda/medicion.py`` y se importa solo en el proceso padre.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    from comun.config import MODES

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=sorted(MODES), required=True)
    ap.add_argument("--tag", required=True, help="identifica la máquina/persona, p. ej. pc1 o pc2")
    ap.add_argument("--reps", type=int, default=3, help="repeticiones medidas por fase")
    ap.add_argument("--epochs", type=int, default=3, help="épocas FIJAS por entrenamiento medido")
    ap.add_argument("--models", nargs="+", default=["cnn1", "cnn2", "mlp"], choices=["cnn1", "cnn2", "mlp"])
    ap.add_argument("--limit", type=int, default=None, help="usar solo N imágenes al azar (pruebas rápidas)")
    ap.add_argument("--amp", action="store_true", help="precisión mixta FP16 en GPU (variante opcional)")
    ap.add_argument("--adam", choices=["torch", "numba"], default="torch",
                    help="paso de Adam: torch.optim (oficial) o kernel CUDA propio de Numba (variante; usa otro --tag)")
    ap.add_argument("--no-monitor", action="store_true", help="no registrar el log de recursos")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.adam == "numba":
        # nvJitLink 12.9 debe cargarse ANTES que torch (ver kernel_adam_numba.py)
        from cuda.bindings import nvjitlink

        nvjitlink.version()
    from paralelo_cuda.medicion import run  # import pesado (torch) solo en el proceso padre

    run(args)
