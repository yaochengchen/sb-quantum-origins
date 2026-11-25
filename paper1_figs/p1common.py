"""Shared helpers for the MSBQA paper-1 figure scripts (read existing results only, no simulation).

All scripts are run from msbqa_phase1/ (or anywhere: paths are resolved relative to this file):
    python paper1_figs/fig2_who_wins.py
Each script writes paper1_figs/<name>.pdf, .png and <name>_numbers.txt (the numbers quoted in the paper).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # msbqa_phase1/
RUN = os.path.join(ROOT, "run_v4")
sys.path.insert(0, os.path.join(ROOT, "code"))   # kpo_common


def P(*a):
    return os.path.join(ROOT, *a)


# column names of run_v4/phase1b/per_instance_1b.csv
C_L2G_GOTO = "L2g eps=1e-6 (Goto 2016 classical)"
C_L2G = "L2g eps*=0.03"
C_L3 = "L3 eps*=0.001"
C_L4 = "L4 eps*=1e-06"
C_BSB = "bSB* (c 0.5; dt 1; 10000 steps; A 0.001)"
C_DSB = "dSB* (c 0.5; dt 0.5; 10000 steps; A 0.001)"

BLUE, RED, ORANGE, GREY, GREEN, PURPLE = "#2B6CB0", "#C0392B", "#E08E0B", "#7F7F7F", "#2E8B57", "#7B4FA0"


class Numbers:
    """Collects the printed numbers and writes them next to the figure."""

    def __init__(self, name):
        self.name = name
        self.lines = []

    def __call__(self, s=""):
        print(s)
        self.lines.append(str(s))

    def write(self):
        with open(os.path.join(HERE, f"{self.name}_numbers.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.lines) + "\n")


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # the font is fixed (DejaVu Sans ships with matplotlib), so every figure looks the same on every machine
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def panel_labels(axes, x=-0.02, y=1.02):
    """(a), (b), ... just above the top-left corner of each axes (outside the data area)."""
    for a, t in zip(np.ravel(axes), "abcdefghij"):
        a.text(x, y, f"({t})", transform=a.transAxes, va="bottom", ha="left", fontweight="bold")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    print("written:", os.path.join(HERE, name + ".pdf"))


def load_1b():
    """Test set G4 0-299: per-instance quantum and classical results (phase 1b) + crossing flags (weyl)."""
    df = pd.read_csv(P("run_v4", "phase1b", "per_instance_1b.csv")).set_index("inst")
    br = pd.read_csv(P("run_v4", "branches", "branches_weyl.csv")).set_index("inst")
    df["cross"] = br["cross"].reindex(df.index).clip(lower=0).fillna(0).astype(int)
    for c in ("p_c", "ham_c", "xsq_c"):
        df[c] = br[c].reindex(df.index)
    df["cls"] = np.where(df["cross"] == 1, "crossing", np.where(df["trap"] == 1, "trap, no crossing", "other"))
    return df


def load_branches():
    return pd.read_csv(P("run_v4", "branches", "branches_weyl.csv")).set_index("inst")


CLS_STYLE = {"other": dict(c=BLUE, s=6, alpha=0.45, z=1),
             "trap, no crossing": dict(c=ORANGE, s=12, alpha=0.85, z=2),
             "crossing": dict(c=RED, s=14, alpha=0.9, z=3)}
