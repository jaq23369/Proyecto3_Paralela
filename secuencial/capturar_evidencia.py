"""Captura de pantalla de UNA ventana (no de toda la pantalla) para la evidencia de ejecución.

Se usa para fotografiar el Administrador de Tareas (u otro monitor de recursos) mientras corre la medición. Solo se
captura el rectángulo de la ventana indicada, de modo que no aparezca nada más del escritorio.

Uso::  python secuencial/capturar_evidencia.py --lista
       python secuencial/capturar_evidencia.py --titulo "Task Manager" --salida results/evidencia/prueba.png
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import sys
import time
from pathlib import Path

from PIL import ImageGrab

user32 = ctypes.windll.user32
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)   # coordenadas reales en pantallas con escala > 100 %
except Exception:
    pass


def ventanas_visibles() -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                out.append((hwnd, buf.value))
        return True

    user32.EnumWindows(cb, 0)
    return out


def buscar(titulo: str) -> int | None:
    for hwnd, t in ventanas_visibles():
        if titulo.lower() in t.lower():
            return hwnd
    return None


def rect(hwnd: int) -> tuple[int, int, int, int]:
    """Rectángulo VISIBLE de la ventana (sin los bordes invisibles de redimensionado que Windows incluye)."""
    r = wt.RECT()
    DWMWA_EXTENDED_FRAME_BOUNDS = 9
    if ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(r), ctypes.sizeof(r)) != 0:
        user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def capturar(hwnd: int, salida: Path) -> tuple[int, int]:
    l, t, r, b = rect(hwnd)
    img = ImageGrab.grab(bbox=(l, t, r, b), all_screens=True)
    salida.parent.mkdir(parents=True, exist_ok=True)
    img.save(salida)
    return img.size


def fase_actual(log: Path | None) -> str:
    """Fase que está midiendo el benchmark en este instante (última fila del log de recursos)."""
    try:
        ultima = log.read_text(encoding="utf-8").strip().splitlines()[-1].split(",")
        return f"{ultima[1]}"
    except Exception:
        return "sin_log"


def bucle(hwnd: int, carpeta: Path, cada: float, duracion: float, log: Path | None) -> None:
    """Toma una captura cada `cada` s durante `duracion` s; el nombre incluye el tiempo y la fase en curso."""
    HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE, SWP_NOACTIVATE = -1, 0x2, 0x1, 0x10
    t0, n = time.time(), 0
    while time.time() - t0 < duracion:
        user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)  # mantenerla encima
        t = time.time() - t0
        capturar(hwnd, carpeta / f"frame_{n:03d}_t{t:05.1f}s_{fase_actual(log)}.png")
        n += 1
        time.sleep(cada)
    print(f"{n} capturas en {carpeta}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lista", action="store_true", help="lista las ventanas visibles")
    ap.add_argument("--titulo", help="texto que debe contener el título de la ventana")
    ap.add_argument("--salida", type=Path)
    ap.add_argument("--bucle", action="store_true", help="capturas periódicas durante una medición")
    ap.add_argument("--carpeta", type=Path, help="(bucle) carpeta de salida")
    ap.add_argument("--cada", type=float, default=2.0, help="(bucle) segundos entre capturas")
    ap.add_argument("--duracion", type=float, default=120.0, help="(bucle) segundos totales")
    ap.add_argument("--log", type=Path, help="(bucle) log de recursos de la medición, para etiquetar la fase")
    a = ap.parse_args()
    if a.lista:
        for hwnd, t in ventanas_visibles():
            print(hwnd, "|", t, "|", rect(hwnd))
        return
    hwnd = buscar(a.titulo)
    if not hwnd:
        sys.exit(f"No hay ninguna ventana visible con '{a.titulo}'")
    if a.bucle:
        bucle(hwnd, a.carpeta, a.cada, a.duracion, a.log)
    else:
        print("capturado", capturar(hwnd, a.salida), "->", a.salida)


if __name__ == "__main__":
    main()
