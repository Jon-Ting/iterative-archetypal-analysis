"""Plot time-complexity results for the manuscript figures.

Reads runtime files produced by ``runTC_RAA.py`` and ``runTC_IAA.py`` and
generates four figures:

1. ``TC_RAA.png``         — RAA duration vs number of samples (linear scale).
2. ``TC_RAA_loglog.png``  — Same, with O(n) reference line, log-log scale.
3. ``TC_PIAA.png``        — IAA duration vs number of samples.
4. ``TC_PIAA_loglog.png`` — Same, with O(n) reference line, log-log scale.
5. ``TC_RAAPIAA.png``     — RAA vs IAA comparison (linear scale).
6. ``TC_RAAPIAA_loglog.png`` — RAA vs IAA comparison with O(n) references.

Input directory: ``data/runTimes/computationalComplexity/``
Output directory: (same as script location, i.e. ``iaa-complexity-scaling/``)

Usage
-----
    python plotTC.py
"""

from math import ceil
from os import listdir

import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Plot settings (match the notebook)
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

TC_NUM_SAMPLES = (100, 200, 400, 800, 1600, 3200)
INPUT_DIR = "data/runTimes/computationalComplexity"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_durations(keyword: str) -> list[list[float]]:
    """Return a list of duration lists, one inner list per matching file."""
    fnames = natsorted([f for f in listdir(INPUT_DIR) if keyword in f])
    all_durations = []
    for fname in fnames:
        print(f"  Loading {fname}")
        durations = []
        with open(f"{INPUT_DIR}/{fname}") as fh:
            for line in fh:
                if "Run " in line:
                    # "Run {i}: {duration:.3f} s" or
                    # "Run {i}: {duration:.3f} s  EV: ..."
                    parts = line.strip().split()
                    durations.append(float(parts[2]))
        all_durations.append(durations)
    return all_durations


def remove_outliers_iqr(df: pd.DataFrame) -> list[float]:
    """Return IQR-filtered column means."""
    means = []
    for col in df.columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        clean = df[col].loc[(df[col] >= lo) & (df[col] <= hi)]
        print(f"  {col} samples: retained {len(clean)} / {len(df)}")
        means.append(clean.mean())
    return means


def box_plots(df: pd.DataFrame, x_label: str, method: str,
              analysis_type: str) -> None:
    """Save a grid of box plots, one panel per column."""
    col_num = ceil(len(df.columns) / 2)
    row_num = 2
    fig, ax = plt.subplots(row_num, col_num,
                           figsize=(1.5 * col_num, 0.9 * col_num), dpi=DPI)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    row = 0
    palette = sns.color_palette("Paired", row_num * col_num)
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Loading RAA runtimes ===")
    raa_raw = load_durations("TC_RAA_")
    raa_df = pd.DataFrame(np.array(raa_raw).T, columns=TC_NUM_SAMPLES)
    box_plots(raa_df, "Number of samples", "RAA", "TC")
    raa_durations = remove_outliers_iqr(raa_df)
    raa_df_clean = pd.DataFrame({"numSample": TC_NUM_SAMPLES,
                                  "duration": raa_durations})

    print("\n=== Loading IAA runtimes ===")
    piaa_raw = load_durations("TC_PIAA_")
    piaa_df = pd.DataFrame(np.array(piaa_raw).T, columns=TC_NUM_SAMPLES)
    box_plots(piaa_df, "Number of samples", "PIAA", "TC")
    piaa_durations = remove_outliers_iqr(piaa_df)
    piaa_df_clean = pd.DataFrame({"numSample": TC_NUM_SAMPLES,
                                   "duration": piaa_durations})

    # ------------------------------------------------------------------
    # 1. TC_RAA.png — linear scale
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    plt.plot(raa_df_clean["numSample"], raa_df_clean["duration"],
             linewidth=LINE_WIDTH, linestyle="--", marker="s",
             markersize=MARKER_SIZE, color=HEX_CODES[1], zorder=3)
    plt.xlabel("Number of samples", fontsize="medium")
    plt.ylabel("Duration (s)", fontsize="medium")
    plt.tick_params(axis="both", labelsize="small")
    plt.grid(linestyle="dotted")
    plt.tight_layout()
    plt.savefig("TC_RAA.png")
    plt.close()
    print("Saved TC_RAA.png")

    # ------------------------------------------------------------------
    # 2. TC_RAA_loglog.png — log-log with O(n) reference
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    n0 = raa_df_clean["numSample"].iloc[0]
    t0 = raa_df_clean["duration"].iloc[0]
    plt.plot(raa_df_clean["numSample"], raa_df_clean["duration"],
             linewidth=LINE_WIDTH, linestyle="--", marker="s",
             markersize=MARKER_SIZE, color=HEX_CODES[1], zorder=3)
    plt.plot(raa_df_clean["numSample"],
             t0 * (raa_df_clean["numSample"] / n0),
             linewidth=LINE_WIDTH, linestyle="-", marker=None,
             markersize=MARKER_SIZE, color=HEX_CODES[0], zorder=2)
    plt.loglog()
    plt.xlabel("Number of samples", fontsize="medium")
    plt.ylabel("Duration (s)", fontsize="medium")
    plt.tick_params(axis="both", labelsize="small")
    plt.grid(linestyle="dotted")
    plt.legend(labels=("Actual", r"$O(n)$ Ideal"), fontsize="small")
    plt.tight_layout()
    plt.savefig("TC_RAA_loglog.png")
    plt.close()
    print("Saved TC_RAA_loglog.png")

    # ------------------------------------------------------------------
    # 3. TC_PIAA.png — linear scale
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    plt.plot(piaa_df_clean["numSample"], piaa_df_clean["duration"],
             linewidth=LINE_WIDTH, linestyle="--", marker="s",
             markersize=MARKER_SIZE, color=HEX_CODES[1], zorder=3)
    plt.xlabel("Number of samples", fontsize="medium")
    plt.ylabel("Duration (s)", fontsize="medium")
    plt.tick_params(axis="both", labelsize="small")
    plt.grid(linestyle="dotted")
    plt.tight_layout()
    plt.savefig("TC_PIAA.png")
    plt.close()
    print("Saved TC_PIAA.png")

    # ------------------------------------------------------------------
    # 4. TC_PIAA_loglog.png — log-log with O(n) reference
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    n0 = piaa_df_clean["numSample"].iloc[0]
    t0 = piaa_df_clean["duration"].iloc[0]
    plt.plot(piaa_df_clean["numSample"], piaa_df_clean["duration"],
             linewidth=LINE_WIDTH, linestyle="--", marker="s",
             markersize=MARKER_SIZE, color=HEX_CODES[1], zorder=3)
    plt.plot(piaa_df_clean["numSample"],
             t0 * (piaa_df_clean["numSample"] / n0),
             linewidth=LINE_WIDTH, linestyle="-", marker=None,
             markersize=MARKER_SIZE, color=HEX_CODES[0], zorder=2)
    plt.loglog()
    plt.xlabel("Number of samples", fontsize="medium")
    plt.ylabel("Duration (s)", fontsize="medium")
    plt.tick_params(axis="both", labelsize="small")
    plt.grid(linestyle="dotted")
    plt.legend(labels=("Actual", r"$O(n)$ Ideal"), fontsize="small")
    plt.tight_layout()
    plt.savefig("TC_PIAA_loglog.png")
    plt.close()
    print("Saved TC_PIAA_loglog.png")

    # ------------------------------------------------------------------
    # 5. TC_RAAPIAA.png — comparison, linear scale
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(raa_df_clean["numSample"], raa_df_clean["duration"],
            linewidth=LINE_WIDTH, linestyle="--", marker="s",
            markersize=MARKER_SIZE, color=HEX_CODES[1], zorder=3)
    ax.plot(piaa_df_clean["numSample"], piaa_df_clean["duration"],
            linewidth=LINE_WIDTH, linestyle="--", marker="x",
            markersize=MARKER_SIZE, color=HEX_CODES[2], zorder=3)
    ax.set_xlabel("Number of samples", fontsize="medium")
    ax.set_ylabel("Duration (s)", fontsize="medium")
    ax.tick_params(axis="both", labelsize="small")
    ax.grid(linestyle="dotted")
    ax.legend(labels=("RAA", "IAA"), fontsize="small")
    plt.tight_layout()
    plt.savefig("TC_RAAPIAA.png")
    plt.close()
    print("Saved TC_RAAPIAA.png")

    # ------------------------------------------------------------------
    # 6. TC_RAAPIAA_loglog.png — comparison with O(n) references
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    ax = fig.add_subplot(1, 1, 1)
    raa_n0 = raa_df_clean["numSample"].iloc[0]
    raa_t0 = raa_df_clean["duration"].iloc[0]
    piaa_n0 = piaa_df_clean["numSample"].iloc[0]
    piaa_t0 = piaa_df_clean["duration"].iloc[0]

    ax.plot(raa_df_clean["numSample"], raa_df_clean["duration"],
            linewidth=LINE_WIDTH, linestyle="--", marker="s",
            markersize=MARKER_SIZE, color=HEX_CODES[1], zorder=3)
    ax.plot(piaa_df_clean["numSample"], piaa_df_clean["duration"],
            linewidth=LINE_WIDTH, linestyle="--", marker="x",
            markersize=MARKER_SIZE, color=HEX_CODES[2], zorder=3)
    ax.plot(raa_df_clean["numSample"],
            raa_t0 * (raa_df_clean["numSample"] / raa_n0),
            linewidth=LINE_WIDTH, linestyle="-", marker=None,
            markersize=MARKER_SIZE, color=HEX_CODES[0], zorder=2)
    ax.plot(piaa_df_clean["numSample"],
            piaa_t0 * (piaa_df_clean["numSample"] / piaa_n0),
            linewidth=LINE_WIDTH, linestyle="-", marker=None,
            markersize=MARKER_SIZE, color=HEX_CODES[3], zorder=2)
    ax.loglog()
    ax.set_xlabel("Number of samples", fontsize="medium")
    ax.set_ylabel("Duration (s)", fontsize="medium")
    ax.tick_params(axis="both", labelsize="small")
    ax.grid(linestyle="dotted")
    ax.legend(labels=("RAA", "IAA", r"$O(n)$ RAA", r"$O(n)$ IAA"),
              fontsize="small")
    plt.tight_layout()
    plt.savefig("TC_RAAPIAA_loglog.png")
    plt.close()
    print("Saved TC_RAAPIAA_loglog.png")

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
