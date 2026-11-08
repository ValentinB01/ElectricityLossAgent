"""What every script shares: paths, the seed, shared thresholds, and how results are written and drawn."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # The repository root
KELMARSH = ROOT / "data" / "kelmarsh"            # From prepare_data.py
PV_DIR = ROOT / "data" / "pv"                     # From prepare_data.py --pv
FORECASTS = ROOT / "data" / "forecast"            # Last Open-Meteo forecast per site, the fallback when offline
MODELS, RESULTS = ROOT / "models", ROOT / "results"

SEED = 42                                         # CLAUDE.md rule 4
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
STEP_H = 1 / 6                                    # Kelmarsh SCADA: one row per 10 minutes, in hours
CONF_MIN = 0.55                                   # Below this M2 confidence, inspect before sending a repair crew. Assumption.

COLORS = {"blue": "#2a78d6", "orange": "#eb6834", "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e0", "surface": "#fcfcfb"}


def write_result(name: str, res: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))


def chart_style():
    """Shared look for every results chart. Returns pyplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MUTED, GRID, SURFACE = COLORS["muted"], COLORS["grid"], COLORS["surface"]
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                         "axes.titlelocation": "left", "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False})
    return plt


def save_figure(fig, name: str) -> None:
    import matplotlib.pyplot as plt
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / f"{name}.png", dpi=150, facecolor=COLORS["surface"])
    plt.close(fig)


def plot_confusion(ax, labels: list, matrix: list, short: list) -> None:
    """Confusion matrix shaded by share of each true class, with counts written in."""
    cm = np.array(matrix, float)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), short, rotation=30, ha="right", fontsize=9); ax.set_yticks(range(len(labels)), short, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=10, color="white" if share[i, j] > 0.55 else COLORS["ink"])
