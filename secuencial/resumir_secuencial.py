"""Resume la medición de la solución SECUENCIAL (Lab 3 original) en una tabla y gráficos (Paso 2).

Lee lo que produce ``secuencial/medir_secuencial.py`` y NO usa nada de la versión paralela:

  results/secuencial_tabla_<tag>.md                     tiempos por repetición, media ± desviación y tiempo por imagen
  results/figures/secuencial_1_tiempos_<tag>.png        tiempo de cada tarea y qué porcentaje del total representa
  results/figures/secuencial_2_uso_cpu_<tag>.png        cuántos núcleos del CPU trabajan en cada momento

Uso::  python secuencial/resumir_secuencial.py --tag pc1
"""

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

MODEL = {"cnn1": "CNN #1", "cnn2": "CNN #2", "mlp": "MLP"}
GROUP_NAME = {"preprocess": "Preparar las imágenes", "train": "Entrenar", "infer": "Evaluar en test"}
GROUP_COLOR = {"preprocess": "#F2B134", "train": "#2E86AB", "infer": "#7BB661"}
N_IMAGES = {"preprocess": 17400, "train": 12180 * 3, "infer": 2610}   # imágenes procesadas por tarea (train: 3 épocas)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def task_label(phase: str, model: str) -> str:
    return GROUP_NAME[phase] + ("" if model == "-" else f" {MODEL[model]}")


def build_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (phase, model), g in d.groupby(["phase", "model"]):
        reps = g.sort_values("rep")["seconds"].tolist()
        rows.append({"phase": phase, "model": model, "reps": reps, "mean": np.mean(reps), "std": np.std(reps, ddof=1),
                     "ms_per_image": 1000 * np.mean(reps) / N_IMAGES[phase]})
    t = pd.DataFrame(rows)
    order = {"preprocess": 0, "train": 1, "infer": 2}
    return t.sort_values(["phase", "model"], key=lambda c: c.map(order) if c.name == "phase" else c).reset_index(drop=True)


def markdown(t: pd.DataFrame, d: pd.DataFrame, tag: str) -> str:
    n_reps = d["rep"].nunique()
    head = "| " + " | ".join(["Tarea"] + [f"Rep. {i}" for i in range(1, n_reps + 1)] + ["**Media ± desv. (s)**", "ms por imagen"]) + " |"
    lines = [f"### Tiempos de la solución secuencial (Lab 3 original) — {tag}\n", head,
             "|---|" + "---:|" * (n_reps + 2)]
    for r in t.itertuples():
        reps = " | ".join(f"{x:.2f}" for x in r.reps)
        lines.append(f"| {task_label(r.phase, r.model)} | {reps} | **{r.mean:.2f} ± {r.std:.2f}** | {r.ms_per_image:.2f} |")
    totals = d.groupby("rep")["seconds"].sum().sort_index().tolist()
    lines.append("| **TOTAL (todas las tareas)** | " + " | ".join(f"{x:.2f}" for x in totals) +
                 f" | **{np.mean(totals):.2f} ± {np.std(totals, ddof=1):.2f}** | |")
    share = t.groupby("phase")["mean"].sum()
    tot = share.sum()
    lines += ["", "### Reparto del tiempo\n", "| Grupo | Tiempo medio | % del total |", "|---|---:|---:|"]
    for g in ("train", "preprocess", "infer"):
        lines.append(f"| {GROUP_NAME[g]} | {share[g]:.2f} s | {100 * share[g] / tot:.1f} % |")
    lines += ["", "_«ms por imagen» = tiempo medio ÷ imágenes procesadas (preparar: 17,400; entrenar: 12,180 × 3 épocas, e "
              "incluye la validación de cada época; evaluar: 2,610)._"]
    return "\n".join(lines) + "\n"


def fig_tiempos(t: pd.DataFrame, tag: str) -> None:
    s = t.sort_values("mean", ascending=True).reset_index(drop=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.7, 1]})
    y = np.arange(len(s))
    a.barh(y, s["mean"], xerr=s["std"], color=[GROUP_COLOR[p] for p in s["phase"]], capsize=3, height=0.65)
    for i, r in s.iterrows():
        a.text(r["mean"] + r["std"] + 3, i, f"{r['mean']:.1f} s", va="center", fontweight="bold")
    a.set_yticks(y, [task_label(p, m) for p, m in zip(s["phase"], s["model"])])
    a.set_xlim(0, s["mean"].max() * 1.18)
    a.set_xlabel("Segundos (promedio de 3 repeticiones; la línea es la desviación)")
    a.set_title("¿Cuánto tarda cada tarea?", loc="left", fontweight="bold")
    share = t.groupby("phase")["mean"].sum()
    order = ["train", "preprocess", "infer"]
    vals = [share[g] for g in order]
    wedges, _ = b.pie(vals, colors=[GROUP_COLOR[g] for g in order], startangle=90, counterclock=False,
                      wedgeprops=dict(width=0.42, edgecolor="white"))
    b.text(0, 0, f"{sum(vals):.0f} s\nen total", ha="center", va="center", fontweight="bold", fontsize=13)
    b.legend(wedges, [f"{GROUP_NAME[g]}: {100 * v / sum(vals):.1f} %" for g, v in zip(order, vals)],
             loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False)
    b.set_title("¿En qué se va el tiempo?", loc="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"secuencial_1_tiempos_{tag}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_uso_cpu(tag: str) -> bool:
    path = config.LOGS_DIR / f"resources_{tag}_keras_original.csv"
    if not path.exists():
        return False
    d = pd.read_csv(path)
    cores = [c for c in d.columns if c.startswith("cpu") and c.endswith("_pct") and c != "cpu_total_pct"]
    d["nucleos"] = (d[cores] > 50).sum(axis=1).rolling(3, center=True, min_periods=1).median()
    grp = d["phase"].str.replace(r"_r\d+$", "", regex=True).str.replace(r"_(cnn1|cnn2|mlp)", "", regex=True)
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.step(d["t_seconds"], d["nucleos"], where="mid", color="#C1440E", lw=2)
    seg = (grp != grp.shift()).cumsum()
    for _, s in d.groupby(seg):
        g = grp[s.index[0]]
        if g in GROUP_COLOR and len(s) > 1:
            ax.axvspan(s["t_seconds"].iloc[0], s["t_seconds"].iloc[-1], color=GROUP_COLOR[g], alpha=0.18, lw=0)
    ax.set_ylim(0, len(cores) + 0.8)
    ax.set_yticks(range(0, len(cores) + 1, 2))
    ax.set_ylabel(f"Núcleos del CPU ocupados\n(de {len(cores)})")
    ax.set_xlabel("Tiempo (segundos)")
    ax.set_title("¿Cuántos núcleos del CPU trabajan en cada momento?", loc="left", fontweight="bold")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=GROUP_COLOR[g], alpha=0.5, label=GROUP_NAME[g]) for g in ("preprocess", "train", "infer")],
               ncol=3, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(config.FIGURES_DIR / f"secuencial_2_uso_cpu_{tag}.png", dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    f = config.RESULTS_DIR / f"benchmark_{args.tag}_keras_original.csv"
    if not f.exists():
        raise SystemExit(f"Falta {f}. Ejecute primero secuencial/medir_secuencial.py --tag {args.tag}")
    d = pd.read_csv(f)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    t = build_table(d)
    md = markdown(t, d, args.tag)
    (config.RESULTS_DIR / f"secuencial_tabla_{args.tag}.md").write_text(md, encoding="utf-8")
    fig_tiempos(t, args.tag)
    fig_uso_cpu(args.tag)
    print(md)


if __name__ == "__main__":
    main()
