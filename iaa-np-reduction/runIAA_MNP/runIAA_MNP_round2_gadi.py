"""
Execute Round 2 of IAA on MNP with p1=500, p2=5000 on Gadi.

This script expects the 10 subset pickles from Round 1 to already exist in:
  <RUN_DIR>/subsetsOutputsPKLs/MNP10S_randomSplit500Asoutput{1..10}.pkl

It combines them, runs the final AA stage (p2=5000) using 48 threads,
extracts close matches, and saves the final fitted PIAA object.
"""

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

# Ensure iteraa is on python path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "iteraa" / "src"))
from iteraa.piaa import fitPIAA

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
K = 10
P1 = 500
P2 = 5000
N_THREADS = int(os.environ.get("NCPUS", "48"))
RANDOM_STATE = 42
LABEL_COLS = ("Total_E", "Formation_E", "Ele", "eleComb")

DATA_PATH = REPO_ROOT / "reduced-np-regression" / "data" / "MNP_E_DF.pkl"
DATA_NAME = f"MNP{K}S_randomSplit{P1}As"

RUN_DIR = REPO_ROOT / "iaa-np-reduction" / "runIAA_MNP" / "run_p1_500_gadi"
OUTPUTS_DIR = RUN_DIR / "subsetsOutputsPKLs"
RESULTS_DIR = RUN_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t_start = time.time()
    print(f"=== IAA MNP Round 2 (Gadi): k={K}, p1={P1}, p2={P2}, threads={N_THREADS} ===", flush=True)

    # 1. Load data
    print("\n[1/3] Loading + standardising data...", flush=True)
    with open(DATA_PATH, "rb") as f:
        df = pickle.load(f)

    missing = df.isna().any(axis=1)
    if missing.any():
        df = df.loc[~missing]

    drop_cols = [c for c in LABEL_COLS if c in df.columns]
    feats = df.drop(columns=drop_cols).astype(np.float64)
    n, m = feats.shape
    print(f"  Data matrix: n={n}, m={m}", flush=True)

    sc = StandardScaler()
    Xscaled = sc.fit_transform(feats.to_numpy())

    # Save scaler and kept row indices
    with open(RESULTS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(sc, f)
    with open(RESULTS_DIR / "kept_row_indices.pkl", "wb") as f:
        pickle.dump(df.index.tolist(), f)

    # 2. Check subset pickles
    print(f"\n[2/3] Verifying 10 subset pickles in {OUTPUTS_DIR}...", flush=True)
    for i in range(1, K + 1):
        p = OUTPUTS_DIR / f"{DATA_NAME}output{i}.pkl"
        if not p.exists():
            sys.exit(f"Missing subset pickle: {p}")
    print("  All 10 subset pickles present.", flush=True)

    # 3. Round 2 AA
    print(f"\n[3/3] Running Round 2 AA on {K * P1} intermediate archetypes (p2={P2})...", flush=True)
    os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
    t_r2 = time.time()
    AA = fitPIAA(
        Xscaled,
        nArchetypes=P2,
        numSubset=K,
        dataName=DATA_NAME,
        outputsPicklesPath=str(OUTPUTS_DIR),
        shuffle=True,
        randomState=RANDOM_STATE,
        numThreads=N_THREADS,
        verbose=True,
    )
    r2_wall = time.time() - t_r2
    print(f"  Round 2 completed in {r2_wall:.1f} s ({r2_wall / 60:.1f} min)", flush=True)

    print("\nExtracting close matches...", flush=True)
    AA._extractCloseMatch()

    out_file = RESULTS_DIR / f"{DATA_NAME}_p2_{P2}_PIAA.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(AA, f)

    total_time = time.time() - t_start
    print(f"\n=== SUCCESS ===", flush=True)
    print(f"  Explained Variance: {AA.explainedVariance_:.4f}", flush=True)
    print(f"  Saved result to:    {out_file}", flush=True)
    print(f"  Total time:         {total_time:.1f} s ({total_time / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
