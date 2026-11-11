"""실험 스크립트 공통: 손계산 vs 시뮬 비교표, 그림 스타일."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
    "legend.fontsize": 8, "font.family": "DejaVu Sans",
})


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=160)
    plt.close(fig)
    print(f"  → figures/{name}")


def compare(title, rows):
    """rows: (항목, 손계산 값, 시뮬 값).  오차까지 계산해 표로 찍고 최대 오차를 돌려준다."""
    print(f"\n[{title}]")
    print(f"  {'항목':<34}{'손계산':>14}{'시뮬':>14}{'오차':>9}")
    worst = 0.0
    for name, hand, sim in rows:
        err = abs(sim - hand) / abs(hand) if hand else abs(sim)
        worst = max(worst, err)
        pad = 34 - sum(1 for ch in name if ord(ch) > 0x1100)   # 한글은 두 칸
        print(f"  {name:<{pad}}{hand:>14.5g}{sim:>14.5g}{err:>8.2%}")
    return worst
