"""Plot strong-scaling and weak-scaling results.

Reads runtime files produced by ``runSS.py`` and ``runWS.py`` and generates:

Strong scaling (3 figures):
  ``SS_VX_Speedups.png``   — actual speedup vs ideal (log-log)
  ``SS_VX_Speedup.png``    — actual speedup vs Amdahl's law (log-log)
  ``SS_VX_Efficiency.png`` — parallelisation efficiency (log x-axis)

Weak scaling (1 figure):
  ``WS_VX_Efficiency.png`` — parallelisation efficiency (log x-axis)

Input directories:
  ``runtimes/strongScaling/SS_VX_{ncpus}.out``
  ``runtimes/weakScaling/WS_VX_{ncpus}.out``

Usage
-----
    python plotSS_WS.py
"""

from math import ceil

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Plot settings (match complexityScaling.ipynb)
# ---------------------------------------------------------------------------
sns.set_context("paper")
sns.set_palette("Paired")
plt.rc("font", family="sans-serif")
plt.rc("xtick", labelsize="x-small")
plt.rc("ytick", labelsize="x-small")

FIG_SIZE = (3.5, 2.5)
DPI = 300
LINE_WIDTH = 1
MARKER_SIZE = 3
HEX_CODES = ["#E11845", "#0057E9", "#FF00BD", "#8931EF"]

FLIERPROPS = dict(marker="x", markerfacecolor="r", markersize=3,
                  markeredgecolor="k", linestyle="none")
BOXPROPS = dict(linestyle="-", linewidth=0.8, edgecolor="k")
WHISKERPROPS = dict(linestyle="-", linewidth=0.5, color="k")
CAPPROPS = dict(linestyle="-", linewidth=1.2, color="k")
MEDIANPROPS = dict(linestyle="--", linewidth=1.2, color="r")
MEANPROPS = dict(marker="s", markerfacecolor="w", markersize=3,
                 markeredgecolor="b", linestyle="none")

# ---------------------------------------------------------------------------
# Experiment parameters (must match runSS.py / runWS.py)
# ---------------------------------------------------------------------------
SS_NCPUS  = (48, 96, 192, 384, 768, 1536)
SS_NNODES = ( 1,  2,   4,   8,  16,   32)

WS_NCPUS    = (1, 2, 4, 8, 16, 24, 48, 96)
WS_NNODES   = (1, 1, 1, 1,  1,  1,  1,  2)

# Amdahl's law parallel fraction (fitted from original Gadi runs)
AMDAHL_P = 0.21

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_durations_ordered(directory: str, keyword: str,
                            ncpus_ordered: tuple) -> list[list[float]]:
    """Load duration files in the order defined by ncpus_ordered."""
    all_durations = []
    for ncpus in ncpus_ordered:
        fname = f"{keyword}{ncpus}.out"
        fpath = f"{directory}/{fname}"
        durations = []
        with open(fpath) as fh:
            for line in fh:
                if "Run " in line:
                    parts = line.strip().split()
                    durations.append(float(parts[2]))
        all_durations.append(durations)
        print(f"  Loaded {fname}  ({len(durations)} runs)")
    return all_durations


def remove_outliers_iqr(df: pd.DataFrame) -> list[float]:
    """Return IQR-filtered column means."""
    means = []
    for col in df.columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        clean = df[col].loc[(df[col] >= lo) & (df[col] <= hi)]
        print(f"  {col} cpus: retained {len(clean)} / {len(df)}")
        means.append(clean.mean())
    return means


def box_plots(df: pd.DataFrame, x_label: str, method: str,
              analysis_type: str) -> None:
    """Save a grid of box plots."""
    col_num = ceil(len(df.columns) / 2)
    row_num = 2
    fig, ax = plt.subplots(row_num, col_num,
                           figsize=(1.5 * col_num, 0.9 * col_num), dpi=DPI)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    palette = sns.color_palette("Paired", row_num * col_num)
    row = 0
    for i, col_name in enumerate(df.columns):
        col = i - row * col_num
        if col >= col_num:
            row += 1
            col -= col_num
        sns.boxplot(data=df[col_name], ax=ax[row, col], width=0.8,
                    showfliers=True, linewidth=1.2, color=palette[i],
                    showmeans=True, meanline=False, notch=False,
                    flierprops=FLIERPROPS, boxprops=BOXPROPS,
                    whiskerprops=WHISKERPROPS, capprops=CAPPROPS,
                    medianprops=MEDIANPROPS, meanprops=MEANPROPS)
        ax[row, col].set_xlabel(col_name,
                                fontdict={"weight": "bold", "fontsize": "medium"})
        ax[row, col].set_ylabel("")
        ax[row, col].set_xticklabels([])
        ax[row, col].tick_params(axis="y", labelsize="small")
    fig.text(0.5, -0.01, x_label, ha="center", va="center", fontsize="large")
    fig.text(-0.01, 0.5, "Duration (s)", ha="center", va="center",
             rotation="vertical", fontsize="large")
    plt.tight_layout()
    out = f"{analysis_type}_{method}_BoxPlots.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Strong scaling
# ---------------------------------------------------------------------------

def plot_strong_scaling() -> None:
    print("\n=== Strong Scaling ===")
    raw = load_durations_ordered("runtimes/strongScaling", "SS_VX_", SS_NCPUS)
    df = pd.DataFrame(np.array(raw).T, columns=SS_NCPUS)
    box_plots(df, "Number of cores", "VX", "SS")

    durations = remove_outliers_iqr(df)
    ss_df = pd.DataFrame({
        "ncpu": SS_NCPUS,
        "nnode": SS_NNODES,
        "duration": durations,
    })
    ss_df["speedup"] = ss_df["duration"].iloc[0] / ss_df["duration"]
    ss_df["efficiency"] = (
        ss_df["duration"].iloc[0] * ss_df["ncpu"].iloc[0]
    ) / (ss_df["duration"] * ss_df["ncpu"])

    print(ss_df[["ncpu", "nnode", "duration", "speedup", "efficiency"]]
          .to_string(index=False))

    # 1. Speedup vs ideal (log-log)
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ss_df["ncpu"], ss_df["speedup"],
            linewidth=LINE_WIDTH, linestyle="--", marker="o",
            markersize=MARKER_SIZE, color=HEX_CODES[1], zorder=3)
    ax.plot(ss_df["ncpu"], ss_df["ncpu"] / ss_df["ncpu"].iloc[0],
            linewidth=LINE_WIDTH, linestyle="-", marker=None,
            markersize=MARKER_SIZE, color=HEX_CODES[0], zorder=2)
    ax.loglog()
    ax.set_xlabel("Number of cores", fontsize="medium")
    ax.set_ylabel("Speedup", fontsize="medium")
    ax.tick_params(axis="both", labelsize="small")
    ax.grid(linestyle="dotted")
    ax.legend(labels=("Actual", "Ideal"), fontsize="small")
    plt.tight_layout()
    plt.savefig("SS_VX_Speedups.png")
    plt.close()
    print("  Saved SS_VX_Speedups.png")

    # 2. Speedup vs Amdahl's law (log-log)
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    s = np.arange(1, SS_NCPUS[-1] + 1)
    amdahl = 1.0 / ((1 - AMDAHL_P) + AMDAHL_P / s)
    ax.plot(s, amdahl,
            linewidth=LINE_WIDTH, linestyle="-", marker=None,
            markersize=MARKER_SIZE, color=HEX_CODES[0], zorder=2)
    ax.plot(ss_df["ncpu"], ss_df["speedup"],
            linewidth=LINE_WIDTH, linestyle="--", marker="o",
            markersize=MARKER_SIZE, color=HEX_CODES[1], zorder=3)
    ax.loglog()
    ax.set_xlabel("Number of cores", fontsize="medium")
    ax.set_ylabel("Speedup", fontsize="medium")
    ax.tick_params(axis="both", labelsize="small")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.grid(linestyle="dotted")
    ax.legend(labels=("Amdahl's law", "Actual"), fontsize="small")
    plt.tight_layout()
    plt.savefig("SS_VX_Speedup.png")
    plt.close()
    print("  Saved SS_VX_Speedup.png")

    # 3. Efficiency (log x-axis)
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ss_df["ncpu"], ss_df["efficiency"],
            linewidth=LINE_WIDTH, linestyle="--", marker="o",
            markersize=MARKER_SIZE, color=HEX_CODES[1], zorder=3)
    ax.plot(ss_df["ncpu"], [1.0] * len(ss_df),
            linewidth=LINE_WIDTH, linestyle="-", marker=None,
            markersize=MARKER_SIZE, color=HEX_CODES[0], zorder=2)
    plt.xscale("log")
    ax.set_xlabel("Number of cores", fontsize="medium")
    ax.set_ylabel("Efficiency", fontsize="medium")
    ax.tick_params(axis="both", labelsize="small")
    ax.grid(linestyle="dotted")
    ax.legend(labels=("Actual", "Ideal"), fontsize="small")
    plt.tight_layout()
    plt.savefig("SS_VX_Efficiency.png")
    plt.close()
    print("  Saved SS_VX_Efficiency.png")


# ---------------------------------------------------------------------------
# Weak scaling
# ---------------------------------------------------------------------------

def plot_weak_scaling() -> None:
    print("\n=== Weak Scaling ===")
    raw = load_durations_ordered("runtimes/weakScaling", "WS_VX_", WS_NCPUS)
    df = pd.DataFrame(np.array(raw).T, columns=WS_NCPUS)
    box_plots(df, "Number of cores", "VX", "WS")

    durations = remove_outliers_iqr(df)
    ws_df = pd.DataFrame({
        "ncpu": WS_NCPUS,
        "nnode": WS_NNODES,
        "duration": durations,
    })
    # Weak-scaling efficiency = T(1) / T(N) — constant if perfect
    ws_df["efficiency"] = ws_df["duration"].iloc[0] / ws_df["duration"]

    print(ws_df[["ncpu", "nnode", "duration", "efficiency"]]
          .to_string(index=False))

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ws_df["ncpu"], ws_df["efficiency"],
            linewidth=LINE_WIDTH, linestyle="--", marker="o",
            markersize=MARKER_SIZE, color=HEX_CODES[1], zorder=3)
    ax.plot(ws_df["ncpu"], [1.0] * len(ws_df),
            linewidth=LINE_WIDTH, linestyle="-", marker=None,
            markersize=MARKER_SIZE, color=HEX_CODES[0], zorder=2)
    plt.xscale("log")
    ax.set_xlabel("Number of cores", fontsize="medium")
    ax.set_ylabel("Efficiency", fontsize="medium")
    ax.tick_params(axis="both", labelsize="small")
    ax.grid(linestyle="dotted")
    ax.legend(labels=("Actual", "Ideal"), fontsize="small")
    plt.tight_layout()
    plt.savefig("WS_VX_Efficiency.png")
    plt.close()
    print("  Saved WS_VX_Efficiency.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    plot_strong_scaling()
    plot_weak_scaling()
    print("\nAll SS/WS figures saved.")


if __name__ == "__main__":
    main()
