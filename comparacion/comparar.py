"""Compara la versión SECUENCIAL (Lab 3 original) contra la PARALELA (CPU multinúcleo + GPU CUDA).

Lee los CSV que producen ``secuencial/medir_secuencial.py`` y ``paralelo_cuda/medir_cuda.py`` y genera, por cada
``--tag`` (persona/equipo), en ``results/``:

  tabla_comparativa_<tag>.md   tablas de tiempos, speedup por fase y uso de recursos (para pegar en el informe)
  speedup_<tag>.csv            los mismos números en CSV
  figures/1_tiempo_total_<tag>.png        ¿cuánto tarda todo y en qué se va el tiempo?
  figures/2_speedup_por_fase_<tag>.png    ¿cuántas veces más rápido es cada parte?
  figures/3_uso_recursos_secuencial_<tag>.png / ..._paralelo_<tag>.png   ¿quién trabaja en cada momento?

Speedup = tiempo secuencial / tiempo paralelo. Los gráficos muestran SOLO esos dos lados; con
``--con-referencias`` las tablas añaden las referencias opcionales en PyTorch (CPU 1 hilo y CPU multihilo).

Uso::  python comparacion/comparar.py --tag pc1 [--tag pc2] [--con-referencias]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comun import config  # noqa: E402

SEQ, PAR = "keras_original", "cuda"
REFS = ["sequential", "cpu_mt"]
LABEL = {SEQ: "Secuencial (Lab 3 original)", PAR: "Paralelo (CPU multinúcleo + GPU CUDA)",
         "sequential": "PyTorch CPU 1 hilo", "cpu_mt": "PyTorch CPU multihilo"}
PHASE = {"preprocess": "Preparar las imágenes (CPU)", "h2d_transfer": "Copiar los datos a la GPU",
         "train": "Entrenar las redes", "infer": "Evaluar en test"}
COLOR = {"preprocess": "#F2B134", "h2d_transfer": "#8E7DBE", "train": "#2E86AB", "infer": "#7BB661", "total": "#444444"}
MODEL = {"cnn1": "CNN #1", "cnn2": "CNN #2", "mlp": "MLP", "-": ""}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


# ----------------------------------------------------------------------------- datos
def load(tag: str, modes: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for m in modes if (p := config.RESULTS_DIR / f"benchmark_{tag}_{m}.csv").exists()]
    missing = [m for m in (SEQ, PAR) if not (config.RESULTS_DIR / f"benchmark_{tag}_{m}.csv").exists()]
    if missing:
        raise SystemExit(f"Faltan mediciones para '{tag}': {missing}. Ejecute primero secuencial/medir_secuencial.py "
                         "y paralelo_cuda/medir_cuda.py con --tag " + tag)
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    m = df[df["phase"] != "cuda_warmup"]
    out = m.groupby(["mode", "phase", "model"])["seconds"].agg(mean="mean", std="std").reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def totals(df: pd.DataFrame) -> pd.DataFrame:
    m = df[df["phase"] != "cuda_warmup"]
    t = m.groupby(["mode", "rep"])["seconds"].sum().groupby("mode").agg(mean="mean", std="std").reset_index()
    t["std"] = t["std"].fillna(0.0)
    return t


def get(summary: pd.DataFrame, mode: str, phase: str, model: str = "-") -> float:
    r = summary[(summary["mode"] == mode) & (summary["phase"] == phase) & (summary["model"] == model)]
    return float(r["mean"].iloc[0]) if len(r) else np.nan


def phase_sum(summary: pd.DataFrame, mode: str, phase: str) -> float:
    r = summary[(summary["mode"] == mode) & (summary["phase"] == phase)]
    return float(r["mean"].sum()) if len(r) else 0.0


def speedup_rows(summary: pd.DataFrame, tot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase in ("preprocess", "train", "infer"):
        models = summary[(summary["mode"] == SEQ) & (summary["phase"] == phase)]["model"].tolist()
        for model in models:
            s, p = get(summary, SEQ, phase, model), get(summary, PAR, phase, model)
            rows.append({"fase": phase, "modelo": model, "t_secuencial_s": s, "t_paralelo_s": p, "speedup": s / p})
    s, p = (float(tot[tot["mode"] == m]["mean"].iloc[0]) for m in (SEQ, PAR))
    rows.append({"fase": "total", "modelo": "-", "t_secuencial_s": s, "t_paralelo_s": p, "speedup": s / p})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figuras
def fig_tiempo_total(summary: pd.DataFrame, tot: pd.DataFrame, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.9))
    names = [LABEL[SEQ], LABEL[PAR]]
    left = np.zeros(2)
    for phase in ("preprocess", "h2d_transfer", "train", "infer"):
        vals = np.array([phase_sum(summary, SEQ, phase), phase_sum(summary, PAR, phase)])
        ax.barh(names, vals, left=left, color=COLOR[phase], edgecolor="white", label=PHASE[phase], height=0.55)
        for i, v in enumerate(vals):
            if v > 0.05 * max(tot["mean"]):          # solo etiqueta los tramos lo bastante anchos
                ax.text(left[i] + v / 2, i, f"{v:.0f} s", ha="center", va="center", color="white", fontweight="bold")
        left += vals
    for i, mode in enumerate((SEQ, PAR)):
        t = float(tot[tot["mode"] == mode]["mean"].iloc[0])
        ax.text(t + 4, i, f"{t:.0f} s en total", va="center", fontweight="bold")
    speed = float(tot[tot["mode"] == SEQ]["mean"].iloc[0]) / float(tot[tot["mode"] == PAR]["mean"].iloc[0])
    ax.set_title(f"El pipeline completo es {speed:.1f} veces más rápido en paralelo", loc="left", fontweight="bold")
    ax.set_xlabel("Segundos (promedio de 3 repeticiones)")
    ax.set_xlim(0, float(tot["mean"].max()) * 1.25)
    ax.invert_yaxis()
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0.0, -0.2), frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"1_tiempo_total_{tag}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_speedup(rows: pd.DataFrame, tag: str) -> None:
    order = {"preprocess": 0, "train": 1, "infer": 2, "total": 3}
    r = rows.sort_values(["fase", "modelo"], key=lambda c: c.map(order) if c.name == "fase" else c).reset_index(drop=True)
    labels = []
    for x in r.itertuples():
        base = {"preprocess": "Preparar las imágenes", "train": "Entrenar", "infer": "Evaluar en test",
                "total": "TODO EL PIPELINE"}[x.fase]
        labels.append(f"{base} {MODEL[x.modelo]}".strip())
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(r))
    ax.barh(y, r["speedup"], color=[COLOR[f] for f in r["fase"]], height=0.65)
    for i, v in enumerate(r["speedup"]):
        ax.text(v + 0.25, i, f"{v:.1f}×", va="center", fontweight="bold")
    ax.axvline(1, color="black", lw=1, ls="--")
    ax.text(1.15, len(r) - 0.45, "1× = igual de rápido", fontsize=9, va="center")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, float(r["speedup"].max()) * 1.15)
    ax.set_xlabel("Cuántas veces más rápido es el paralelo   (speedup = tiempo secuencial ÷ tiempo paralelo)")
    ax.set_title("¿Cuántas veces más rápido es cada parte?", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"2_speedup_por_fase_{tag}.png", dpi=160)
    plt.close(fig)


def _group(phase: str) -> str:
    g = pd.Series([phase]).str.replace(r"_r\d+$", "", regex=True).str.replace(r"_(cnn1|cnn2|mlp)", "", regex=True)
    return g.iloc[0]


def fig_recursos(tag: str, mode: str) -> None:
    path = config.LOGS_DIR / f"resources_{tag}_{mode}.csv"
    if not path.exists():
        return
    d = pd.read_csv(path)
    cores = [c for c in d.columns if c.startswith("cpu") and c.endswith("_pct") and c != "cpu_total_pct"]
    d["nucleos"] = (d[cores] > 50).sum(axis=1).rolling(3, center=True, min_periods=1).median()
    d["gpu"] = d["gpu_util_pct"].rolling(3, center=True, min_periods=1).mean()
    d["grupo"] = d["phase"].map(_group)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    a1.step(d["t_seconds"], d["nucleos"], where="mid", color="#C1440E", lw=2)
    a1.set_ylim(0, len(cores) + 0.8)
    a1.set_yticks(range(0, len(cores) + 1, 2))
    a1.set_ylabel(f"Núcleos del CPU ocupados\n(de {len(cores)})")
    a1.set_title(f"¿Quién trabaja en cada momento?  ·  {LABEL[mode]}", loc="left", fontweight="bold")
    if mode == PAR:
        a2.plot(d["t_seconds"], d["gpu"], color="#2E86AB", lw=2)
        a2.set_ylabel("Uso de la GPU (%)")
    else:
        a2.plot(d["t_seconds"], np.zeros(len(d)), color="#2E86AB", lw=2)
        a2.text(0.5, 0.55, "La GPU no se usa: TensorFlow en Windows solo corre en CPU", transform=a2.transAxes,
                ha="center", fontsize=11, color="#555555")
        a2.set_ylabel("Uso de la GPU (%)")
    a2.set_ylim(-3, 105)
    a2.set_xlabel("Tiempo (segundos)")
    shade = {"preprocess": COLOR["preprocess"], "train": COLOR["train"], "infer": COLOR["infer"]}
    seg = (d["grupo"] != d["grupo"].shift()).cumsum()
    for _, s in d.groupby(seg):
        g = s["grupo"].iloc[0]
        if g in shade and len(s) > 1:
            for ax in (a1, a2):
                ax.axvspan(s["t_seconds"].iloc[0], s["t_seconds"].iloc[-1], color=shade[g], alpha=0.18, lw=0)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COLOR["preprocess"], alpha=0.5, label="Fase: preparar las imágenes"),
                        Patch(color=COLOR["train"], alpha=0.5, label="Fase: entrenar"),
                        Patch(color=COLOR["infer"], alpha=0.5, label="Fase: evaluar")],
               ncol=3, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    name = "secuencial" if mode == SEQ else "paralelo"
    fig.savefig(config.FIGURES_DIR / f"3_uso_recursos_{name}_{tag}.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- tablas
def resource_table(tag: str) -> str:
    rows = []
    for mode in (SEQ, PAR):
        p = config.LOGS_DIR / f"resources_{tag}_{mode}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        cores = [c for c in d.columns if c.startswith("cpu") and c.endswith("_pct") and c != "cpu_total_pct"]
        # NVML promedia ~1 s: las primeras muestras de cada fase arrastran la fase anterior -> se descartan 1.5 s.
        d["_seg"] = (d["phase"] != d["phase"].shift()).cumsum()
        d = d[d.groupby("_seg").cumcount() >= 3].copy()
        d["grupo"] = d["phase"].map(_group)
        for g, x in d[d["grupo"].isin(["preprocess", "train"])].groupby("grupo"):
            rows.append(f"| {LABEL[mode]} | {PHASE[g]} | {x['cpu_total_pct'].mean():.0f} % | "
                        f"{int((x[cores].mean() > 50).sum())} de {len(cores)} | {x['gpu_util_pct'].mean():.0f} % | "
                        f"{x['gpu_vram_used_mb'].max():.0f} MiB |")
    if not rows:
        return ""
    return ("### Uso de recursos durante cada fase (promedio de los logs de `psutil` y NVML)\n\n"
            "| Versión | Fase | CPU (promedio) | Núcleos con > 50 % de uso | GPU (promedio) | VRAM máxima* |\n"
            "|---|---|---:|---:|---:|---:|\n" + "\n".join(rows) +
            "\n\n\\*La VRAM incluye ≈ 1 GB que ya usa Windows. La inferencia en GPU dura menos que el intervalo de "
            "muestreo (0.5 s) y por eso no aparece.\n")


def markdown(summary, tot, rows, tag, con_refs) -> str:
    modes = [SEQ, PAR] + (REFS if con_refs else [])
    head = "| Fase | Modelo | " + " | ".join(f"{LABEL[m]} (s)" for m in modes) + " | **Speedup** |"
    lines = [f"### Tiempos medios ± desviación estándar (3 repeticiones) — {tag}\n", head,
             "|---|---|" + "---:|" * (len(modes) + 1)]
    def cell(mode, phase, model):
        r = summary[(summary["mode"] == mode) & (summary["phase"] == phase) & (summary["model"] == model)]
        return "—" if r.empty else f"{r['mean'].iloc[0]:.2f} ± {r['std'].iloc[0]:.2f}"
    for x in rows.itertuples():
        if x.fase == "total":
            cells = []
            for m in modes:
                r = tot[tot["mode"] == m]
                cells.append("—" if r.empty else f"**{r['mean'].iloc[0]:.2f} ± {r['std'].iloc[0]:.2f}**")
            lines.append(f"| **TOTAL del pipeline** | | " + " | ".join(cells) + f" | **{x.speedup:.2f}×** |")
        else:
            lines.append(f"| {PHASE[x.fase]} | {MODEL[x.modelo] or '—'} | "
                         + " | ".join(cell(m, x.fase, x.modelo) for m in modes) + f" | {x.speedup:.2f}× |")
    h2d = summary[(summary["mode"] == PAR) & (summary["phase"] == "h2d_transfer")]
    if len(h2d):
        lines.append(f"| {PHASE['h2d_transfer']} | — | " + " | ".join(
            cell(m, "h2d_transfer", "-") for m in modes) + " | (solo existe en paralelo) |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--con-referencias", action="store_true", help="añade las referencias PyTorch CPU a las tablas")
    args = ap.parse_args()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for tag in args.tag:
        df = load(tag, [SEQ, PAR] + REFS)
        summary, tot = summarize(df), totals(df)
        rows = speedup_rows(summary, tot)
        rows.round(4).to_csv(config.RESULTS_DIR / f"speedup_{tag}.csv", index=False)
        md = markdown(summary, tot, rows, tag, args.con_referencias) + "\n" + resource_table(tag)
        warm = df[df["phase"] == "cuda_warmup"]
        if len(warm):
            md += (f"\n_Inicialización única de CUDA (contexto + autotune de cuDNN), medida aparte y **no** incluida "
                   f"en los tiempos: {warm['seconds'].iloc[0]:.1f} s._\n")
        (config.RESULTS_DIR / f"tabla_comparativa_{tag}.md").write_text(md, encoding="utf-8")
        two = [SEQ, PAR]   # los gráficos muestran solo los dos lados; las referencias PyTorch van solo en tablas
        fig_tiempo_total(summary[summary["mode"].isin(two)], tot[tot["mode"].isin(two)], tag)
        fig_speedup(rows, tag)
        fig_recursos(tag, SEQ)
        fig_recursos(tag, PAR)
        print(md)


if __name__ == "__main__":
    main()
