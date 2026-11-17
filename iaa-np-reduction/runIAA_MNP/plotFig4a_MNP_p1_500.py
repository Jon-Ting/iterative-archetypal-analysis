"""
Generate Figure 4a (panel a) simplex plot for the MNP IAA rerun with p1=500.

Run this AFTER runIAA_MNP_local_p1_500.py has completed.

Usage (from repo root):
  iteraa/.venv/bin/python iaa-np-reduction/runIAA_MNP/plotFig4a_MNP_p1_500.py

Output:
  iaa-np-reduction/runIAA_MNP/local_run_p1_500/figs/
    MNP10S_randomSplit_markPrototype_simplex_maxArchs.png   <- panel (a) for Figure 4
    MNP10S_randomSplitAu_markPrototype_simplex_maxArchs.png
    MNP10S_randomSplitPd_markPrototype_simplex_maxArchs.png
    MNP10S_randomSplitPt_markPrototype_simplex_maxArchs.png
"""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "iteraa" / "src"))
from iteraa.plot import createSimplexAx, mapAlfaToSimplex  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = REPO_ROOT / "reduced-np-regression" / "data" / "MNP_E_DF.pkl"
AU_CSV = REPO_ROOT / "reduced-np-regression" / "data" / "raw_csv" / "Au_nanoparticle_dataset.csv"
PD_CSV = REPO_ROOT / "reduced-np-regression" / "data" / "raw_csv" / "Pd_nanoparticle_dataset.csv"
PT_CSV = REPO_ROOT / "reduced-np-regression" / "data" / "raw_csv" / "Pt_nanoparticle_dataset.csv"

RUN_DIR = REPO_ROOT / "iaa-np-reduction" / "runIAA_MNP" / "local_run_p1_500"
RESULTS_DIR = RUN_DIR / "results"
FIGS_DIR = RUN_DIR / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PKL = RESULTS_DIR / "MNP10S_randomSplit500As_p2_5000_PIAA.pkl"
SCALER_PKL = RESULTS_DIR / "scaler.pkl"
KEPT_IDX_PKL = RESULTS_DIR / "kept_row_indices.pkl"

# Plot style
COLOURS = {"Au": "#1f77b4", "Pd": "#ff7f0e", "Pt": "#2ca02c"}
MARKERS = {"Au": "v", "Pd": "D", "Pt": "s"}
FIG_SIZE = (5, 5)
DPI = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_meta_df(kept_indices):
    """
    Reconstruct Ele and D_avg for each kept row in MNP_E_DF.pkl.

    Row ordering in MNP_E_DF.pkl matches concatenation order:
      0–3999   → Au  (4000 rows)
      4000–7999 → Pd  (4000 rows)
      8000–8299 → Pt  (1300 rows)

    D_avg (diameter) is derived as 2 * R_avg (average radius), which
    matches the physical interpretation used for plot marker sizing.
    """
    # Load R_avg from the pickle (kept rows only)
    with open(DATA_PATH, "rb") as f:
        df_full = pickle.load(f)
    df_kept = df_full.loc[kept_indices]

    ele_map = {i: "Au" if i < 4000 else ("Pd" if i < 8000 else "Pt")
               for i in range(9300)}
    ele = pd.Series([ele_map[i] for i in kept_indices], index=kept_indices, name="Ele")
    d_avg = (df_kept["R_avg"] * 2).rename("D_avg")

    return pd.DataFrame({"Ele": ele, "D_avg": d_avg})


def load_artifacts():
    print("Loading PIAA result...", flush=True)
    with open(RESULT_PKL, "rb") as f:
        AA = pickle.load(f)
    with open(SCALER_PKL, "rb") as f:
        sc = pickle.load(f)
    with open(KEPT_IDX_PKL, "rb") as f:
        kept_indices = pickle.load(f)
    print(f"  Archetypes: {AA.nArchetypes}  EV: {AA.explainedVariance_:.4f}", flush=True)
    return AA, sc, kept_indices


def load_xscaled(sc, kept_indices):
    """Re-standardise the feature matrix using the saved scaler."""
    with open(DATA_PATH, "rb") as f:
        df = pickle.load(f)
    df = df.loc[kept_indices]
    drop = [c for c in ("Total_E", "Formation_E", "Ele", "eleComb") if c in df.columns]
    feats = df.drop(columns=drop).astype(float)
    return sc.transform(feats.to_numpy())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_panel(AA, Xscaled, meta, elements=None, suffix="", title=""):
    """
    Draw one simplex panel.

    elements : list of element labels to include (None → all three).
    suffix   : appended to filename before .png
    """
    if elements is None:
        elements = ["Au", "Pd", "Pt"]

    # Map all samples into 2-D simplex coordinates
    mapped_all = mapAlfaToSimplex(AA.alfa, AA)

    # Compute prototype positions (mean of each element group → archetype space)
    proto_positions = {}
    for ele in elements:
        mask = meta["Ele"] == ele
        X_avg = Xscaled[mask].mean(axis=0).reshape(1, -1)
        _, alfa_avg = AA.transform(X_avg)
        proto_positions[ele] = mapAlfaToSimplex(alfa_avg, AA)

    # Create simplex axes
    createSimplexAx(AA, archIDs=range(AA.nArchetypes),
                    showLabel=True, gridLineWidth=0.0, figSize=FIG_SIZE)
    if title:
        plt.title(title, fontsize=9)

    for ele in elements:
        mask = meta["Ele"] == ele
        idxs = np.where(mask)[0]
        plt.scatter(
            mapped_all[idxs, 0],
            mapped_all[idxs, 1],
            c=COLOURS[ele],
            marker=MARKERS[ele],
            edgecolor="k",
            linewidth=0.5,
            s=meta.loc[meta["Ele"] == ele, "D_avg"].values[: len(idxs)] * 10,
            alpha=0.8,
            label=ele,
        )
        plt.scatter(
            proto_positions[ele][:, 0],
            proto_positions[ele][:, 1],
            marker="X",
            s=150,
            color=COLOURS[ele],
            edgecolor="k",
            linewidth=2,
        )

    fname = f"MNP10S_randomSplit{suffix}_markPrototype_simplex_maxArchs.png"
    out_path = FIGS_DIR / fname
    plt.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close()
    print(f"  Saved: {out_path}", flush=True)
    return out_path


def main():
    AA, sc, kept_indices = load_artifacts()
    Xscaled = load_xscaled(sc, kept_indices)
    meta = build_meta_df(kept_indices)
    meta = meta.loc[kept_indices].reset_index(drop=False)  # align with Xscaled rows

    print("Generating plots...", flush=True)

    # Panel (a): all three elements together — this is Figure 4a
    plot_panel(AA, Xscaled, meta, elements=["Au", "Pd", "Pt"],
               suffix="", title="")

    # Individual element panels (for Supporting Information)
    for ele in ["Au", "Pd", "Pt"]:
        plot_panel(AA, Xscaled, meta, elements=[ele], suffix=ele)

    print(f"\nDone. Figures saved to: {FIGS_DIR}", flush=True)
    print("Copy the combined panel to iaa-manuscripts/FIGURES/ to replace Figure 4a.", flush=True)


if __name__ == "__main__":
    main()
