"""Genera los 3 diagramas del Paso 4 (diseño del algoritmo paralelo) en docs/diagramas/.

Estilo formal de informe técnico: fondo blanco, escala de grises y un solo color (azul marino), rectángulos de
línea fina, conectores en ángulo recto y figuras numeradas. Las cifras son de DISEÑO (salen de las dimensiones de las
capas y de la ficha técnica de la GPU), no de una ejecución.

Uso::  python docs/diagramas/generar_diagramas.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
INK, NAVY, MID, LIGHT, PALE = "#1A1A1A", "#1F3864", "#7F7F7F", "#D9D9D9", "#F2F2F2"
_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = "Arial" if "Arial" in _fonts else "DejaVu Sans"


def canvas(w, h, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def rect(ax, x, y, w, h, text="", fc="white", ec=NAVY, lw=1.2, fs=9.5, bold=False, color=INK, ha="center"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))
    if text:
        tx = x + w / 2 if ha == "center" else x + 1.0
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", linespacing=1.35)


def line(ax, pts, color=INK, lw=1.2, arrow=True):
    xs, ys = zip(*pts)
    if arrow:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_capstyle="butt")
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=11, color=color, lw=lw,
                                     shrinkA=0, shrinkB=0))
    else:
        ax.plot(xs, ys, color=color, lw=lw)


def caption(ax, y, text):
    ax.text(50, y, text, ha="center", va="center", fontsize=10.5, style="italic", color=INK)


def lane(ax, y, h, label, header):
    ax.add_patch(Rectangle((8, y), 90, h, fc=PALE, ec=MID, lw=0.9))
    ax.add_patch(Rectangle((2, y), 6, h, fc=NAVY, ec=NAVY))
    ax.text(5, y + h / 2, label, rotation=90, ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.text(96.5, y + h - 2.1, header, ha="right", va="center", fontsize=10.5, fontweight="bold", color=NAVY)


# ------------------------------------------------------------------------------------------------ Figura 1
def figura_flujo():
    fig, ax = canvas(13, 6.4, 50)
    lane(ax, 28, 21, "CPU", "Fase A. Preparación de las imágenes  (12 procesos en paralelo; 272 bloques de 64 imágenes)")
    cw, cy, ch = 25, 30.2, 10
    cx = [11, 39, 67]
    a_txt = ["Lectura y decodificación\ndel JPG (200 × 200 × 3)\n[Ops. 2–3]",
             "Conversión a RGB y\nreducción a 64 × 64\n[Ops. 4–5]",
             "Normalización (÷ 255) y apilado\nTensor 17,400 × 3 × 64 × 64\n[Ops. 6–7]"]
    for x, t in zip(cx, a_txt):
        rect(ax, x, cy, cw, ch, t, fs=9.3)
    for i in range(2):
        line(ax, [(cx[i] + cw, cy + ch / 2), (cx[i + 1], cy + ch / 2)])

    lane(ax, 2, 21, "GPU", "Fase B. Entrenamiento y evaluación  (CUDA)")
    gw, gy, gh = 20, 4.2, 10
    gx = [11, 33, 55, 77]
    b_txt = ["Datos residentes en la\nmemoria de la GPU\n(816 MiB de 6 GB)",
             "Entrenamiento por lotes\nde 64 imágenes\n[Ops. 9–14]  191 lotes por época",
             "Evaluación en\nvalidación, cada época\n[Op. 15]",
             "Predicción en prueba\ny métricas\n[Op. 16]"]
    for x, t in zip(gx, b_txt):
        rect(ax, x, gy, gw, gh, t, fs=9.0)
    for i in range(3):
        line(ax, [(gx[i] + gw, gy + gh / 2), (gx[i + 1], gy + gh / 2)])

    # conector en ángulo recto CPU -> GPU
    xs = cx[2] + cw / 2
    line(ax, [(xs, cy), (xs, 25.5), (gx[0] + gw / 2, 25.5), (gx[0] + gw / 2, gy + gh)])
    ax.text(gx[0] + gw / 2 + 2, 26.4, "Transferencia CPU → GPU, una sola vez (≈ 816 MiB)", fontsize=9.3, va="bottom",
            ha="left", color=INK)
    caption(ax, -1.0, "Figura 1. Flujo general del algoritmo paralelo propuesto  ([Op.] = número en la Tabla 1 del Paso 1)")
    fig.savefig(OUT / "diagrama_1_flujo_general.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ Figura 2
def figura_paso():
    fig, ax = canvas(13.5, 5.6, 47)
    names =["1. Lote", "2. Adelante", "3. Pérdida", "4. Atrás", "5. Actualización (Adam)"]
    ops = ["—", "9, 10, 11", "12", "13", "14"]
    desc = ["64 imágenes\ntomadas de la\nmemoria de la GPU",
            "Convolución + ReLU,\nBatchNorm, pooling,\ncapas densas",
            "Softmax y\nentropía cruzada",
            "Gradiente de cada\ncapa, de la salida\na la entrada",
            "Actualiza los\n2,245,437 pesos"]
    par = ["Por índice\n(sin copia desde el CPU)", "1 hilo por valor de salida\n(8,388,608 en la capa mayor)",
           "1 bloque por imagen\n64 bloques × 32 hilos", "Misma geometría\nque «Adelante»", "1 hilo por peso\n8,772 bloques × 256 hilos"]
    x0, w, gap = 16, 15.4, 1.5
    # filas (de abajo hacia arriba): reparto 3–13.5 · operaciones 13.5–20.5 · qué hace 20.5–31.5 · paso 31.5–39.5
    rect(ax, 2, 31.5, 13, 8, "Paso", fc=NAVY, ec=NAVY, color="white", bold=True, fs=9.5)
    rect(ax, 2, 20.5, 13, 11, "Qué hace", fc=PALE, ec=MID, bold=True, fs=9)
    rect(ax, 2, 13.5, 13, 7, "Operaciones\n(Tabla 1)", fc=PALE, ec=MID, bold=True, fs=9)
    rect(ax, 2, 3, 13, 10.5, "Reparto de\nhilos y bloques\nen la GPU", fc=PALE, ec=MID, bold=True, fs=9)
    for i in range(5):
        x = x0 + i * (w + gap)
        rect(ax, x, 31.5, w, 8, names[i], fc=NAVY, ec=NAVY, color="white", bold=True, fs=9.3)
        rect(ax, x, 20.5, w, 11, desc[i], fc="white", ec=MID, fs=8.8)
        rect(ax, x, 13.5, w, 7, ops[i], fc="white", ec=MID, fs=10)
        rect(ax, x, 3, w, 10.5, par[i], fc="white", ec=MID, fs=8.6)
        if i < 4:
            line(ax, [(x + w, 35.5), (x + w + gap, 35.5)], lw=1.4)
    # bucle (sobre los encabezados, sin cruzar texto)
    xr = x0 + 4 * (w + gap) + w / 2
    xl = x0 + w / 2
    line(ax, [(xr, 39.5), (xr, 43.2), (xl, 43.2), (xl, 39.6)], color=NAVY, lw=1.2)
    ax.text((xr + xl) / 2, 44.0, "Se repite con el siguiente lote: 191 veces por época", ha="center", va="bottom", fontsize=9.3,
            color=NAVY)
    caption(ax, -0.6, "Figura 2. Un paso de entrenamiento en la GPU: orden de ejecución y reparto del trabajo "
                      "(los pasos son secuenciales; el paralelismo está dentro de cada uno)")
    fig.savefig(OUT / "diagrama_2_paso_de_entrenamiento.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ Figura 3
def figura_hilos():
    fig, ax = canvas(13.5, 6.9, 54)
    ax.text(50, 52.4, "Ejemplo: 2.ª capa convolucional de la CNN #2, lote de 64 imágenes, 32 filtros de 3 × 3", ha="center",
            fontsize=10, color=INK)

    # encabezados de los tres paneles
    panels = [(2, 30, "GRID  (toda la capa)"), (35, 30, "BLOQUE  (una tesela)"), (68, 30, "HILO  (un valor de salida)")]
    for x, w, t in panels:
        rect(ax, x, 44, w, 5, t, fc=NAVY, ec=NAVY, color="white", bold=True, fs=10)
        ax.add_patch(Rectangle((x, 17), w, 27, fc="white", ec=MID, lw=0.9))

    # --- Grid: pila de planos
    gx, gy, s = 6, 24, 15
    for k in range(4, 0, -1):
        ax.add_patch(Rectangle((gx + k * 1.2, gy + k * 1.2), s, s, fc=PALE, ec=MID, lw=0.8))
    ax.add_patch(Rectangle((gx, gy), s, s, fc="white", ec=INK, lw=1.0))
    for i in range(1, 4):
        ax.plot([gx + i * s / 4] * 2, [gy, gy + s], color=MID, lw=0.8)
        ax.plot([gx, gx + s], [gy + i * s / 4] * 2, color=MID, lw=0.8)
    tx0, ty0 = gx + 2 * s / 4, gy + s / 4                      # tesela resaltada
    ax.add_patch(Rectangle((tx0, ty0), s / 4, s / 4, fc=NAVY, ec=NAVY))
    ax.text(17, 20.2, "2,048 planos de 64 × 64 (64 imágenes × 32 filtros)\ncada plano = 4 × 4 teselas de 16 × 16",
            ha="center", va="center", fontsize=8.2, linespacing=1.4)

    # --- Bloque: malla 16x16
    bx, by, n, c = 41, 24, 16, 0.95
    ax.add_patch(Rectangle((bx, by), n * c, n * c, fc="white", ec=INK, lw=1.0))
    for i in range(1, n):
        ax.plot([bx + i * c] * 2, [by, by + n * c], color=LIGHT, lw=0.6)
        ax.plot([bx, bx + n * c], [by + i * c] * 2, color=LIGHT, lw=0.6)
    ax.add_patch(Rectangle((bx + 9 * c, by + 6 * c), c, c, fc=NAVY, ec=NAVY))
    ax.text(bx + n * c / 2, 20.2, "16 × 16 = 256 hilos", ha="center", va="center", fontsize=8.8)

    # --- Hilo
    rect(ax, 70, 27, 26, 11, "salida[imagen, filtro, y, x]\n=  Σ  entrada × peso\nsobre 3 × 3 × 32 = 288 productos", fs=9, ec=INK, lw=1.0)
    ax.text(83, 22, "Los hilos son independientes entre sí:\nse ejecutan a la vez", ha="center", va="center", fontsize=8.8,
            linespacing=1.3)

    # conectores entre paneles (acercamiento)
    y1 = ty0 + s / 8                                              # centro de la tesela resaltada
    line(ax, [(tx0 + s / 4, y1), (bx, y1)], color=NAVY, lw=1.0)
    y2 = by + 6.5 * c                                             # centro de la celda (hilo) resaltada
    line(ax, [(bx + 10 * c, y2), (70, y2)], color=NAVY, lw=1.0)

    # tabla de parámetros
    rows = [("Hilos por bloque", "16 × 16 = 256"),
            ("Bloques del grid", "4 × 4 × 2,048 = 32,768"),
            ("Hilos en total", "32,768 × 256 = 8,388,608"),
            ("Hilos simultáneos de la GPU (RTX 2060)", "30 SM × 1,024 = 30,720   →   ≈ 273 oleadas")]
    ty, rh = 13.5, 3.2
    rect(ax, 2, ty, 96, rh, "Parámetros del diseño", fc=NAVY, ec=NAVY, color="white", bold=True, fs=9.5)
    for i, (a, b) in enumerate(rows):
        y = ty - (i + 1) * rh
        rect(ax, 2, y, 46, rh, a, fc=PALE, ec=MID, fs=9, ha="left")
        rect(ax, 48, y, 50, rh, b, fc="white", ec=MID, fs=9, ha="left")
    caption(ax, -1.2, "Figura 3. Distribución de hilos, bloques y grid en una convolución")
    fig.savefig(OUT / "diagrama_3_hilos_bloques_grid.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    figura_flujo()
    figura_paso()
    figura_hilos()
    print("diagramas generados en", OUT)
