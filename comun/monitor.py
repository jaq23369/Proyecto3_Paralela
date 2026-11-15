"""Monitor de recursos para la evidencia de ejecución (CPU por núcleo, RAM, GPU).

Muestrea en un hilo aparte cada ``interval`` segundos y escribe un CSV con:
uso de cada núcleo lógico del CPU, RAM, y de la GPU (utilización de SMs,
utilización de memoria, VRAM usada, potencia y temperatura vía NVML, la misma
fuente que ``nvidia-smi``). La columna ``phase`` se actualiza con ``mark()`` para
poder recortar el log por fase al graficar.

Uso::

    with ResourceMonitor("logs/resources_pc1_paralelo.csv") as mon:
        mon.mark("train_cnn2_gpu")
        ...
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML = True
except Exception:  # sin driver NVIDIA / sin nvidia-ml-py
    _NVML = False


class ResourceMonitor:
    def __init__(self, csv_path: str | Path, interval: float = 0.5, gpu_index: int = 0):
        self.csv_path = Path(csv_path)
        self.interval = interval
        self._phase = "idle"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index) if _NVML else None
        self.n_cpu = psutil.cpu_count(logical=True)

    def mark(self, phase: str) -> None:
        self._phase = phase

    def _sample_gpu(self) -> list:
        if not self._handle:
            return ["", "", "", "", ""]
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
        return [util.gpu, util.memory, round(mem.used / 2**20), round(power, 1), temp]

    def _run(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        psutil.cpu_percent(percpu=True)  # primera lectura descartable
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["t_seconds", "phase", "cpu_total_pct", *[f"cpu{i}_pct" for i in range(self.n_cpu)],
                 "ram_used_mb", "gpu_util_pct", "gpu_mem_util_pct", "gpu_vram_used_mb",
                 "gpu_power_w", "gpu_temp_c"]
            )
            while not self._stop.is_set():
                per_core = psutil.cpu_percent(interval=self.interval, percpu=True)
                writer.writerow(
                    [round(time.perf_counter() - t0, 2), self._phase,
                     round(sum(per_core) / len(per_core), 1), *per_core,
                     round(psutil.virtual_memory().used / 2**20), *self._sample_gpu()]
                )
                f.flush()

    def __enter__(self) -> "ResourceMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
