"""
Run IAA on the MNP dataset locally.
Configuration: k=10, p1=500, p2=5000, random_state=42

Round 1: 10 subset AAs run in parallel (one process per subset, 1 SPAMS thread each).
Round 2: final AA on the 10*500=5000 intermediate archetypes, using all CPU threads.

Output: iaa-np-reduction/runIAA_MNP/local_run_p1_500/results/
  MNP10S_randomSplit500As_p2_5000_PIAA.pkl  — fitted PIAA object
  scaler.pkl                                 — fitted StandardScaler
  kept_row_indices.pkl                       — original DF row indices after NaN drop
Run time estimate: ~6 min (Round 1) + ~4–8 h (Round 2).

Usage (from repo root):
  nohup iteraa/.venv/bin/python iaa-np-reduction/runIAA_MNP/runIAA_MNP_local_p1_500.py \
      > iaa-np-reduction/runIAA_MNP/local_run_p1_500/run.log 2>&1 &
"""

# ---------------------------------------------------------------------------
# Pin BLAS/OpenMP to 1 thread BEFORE numpy is imported.
# Round-1 worker processes inherit these env vars via fork().
# ---------------------------------------------------------------------------
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "iteraa" / "src"))
from iteraa.piaa import fitPIAA  # noqa: E402  (import after sys.path change)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
K = 10            # number of subsets
P1 = 500          # archetypes per subset (round 1)
P2 = 5000         # final archetypes (round 2)
N_THREADS_R2 = 14  # SPAMS threads for round 2 (all logical CPUs on Ultra 7 165U)
RANDOM_STATE = 42
LABEL_COLS = ("Total_E", "Formation_E", "Ele", "eleComb")
DATA_PATH = REPO_ROOT / "reduced-np-regression" / "data" / "MNP_E_DF.pkl"
DATA_NAME = f"MNP{K}S_randomSplit{P1}As"

RUN_DIR = REPO_ROOT / "iaa-np-reduction" / "runIAA_MNP" / "local_run_p1_500"
OUTPUTS_DIR = RUN_DIR / "subsetsOutputsPKLs"   # round-1 output pickles
RESULTS_DIR = RUN_DIR / "results"

for _d in [OUTPUTS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Worker (must be module-level for ProcessPoolExecutor on Linux fork)
# ---------------------------------------------------------------------------
def _subset_worker(args):
    """Run AA on one data subset and write the output pickle."""
    import os
    # Re-pin inside the worker to be explicit (inherited, but be safe)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"
    import time
    import pickle
    import numpy as np
    import spams

    subset_idx, sample_idxs, X_subset_T, outputs_dir, data_name, p1 = args
    t0 = time.time()
    Z, A, B = spams.archetypalAnalysis(
        np.asfortranarray(X_subset_T),  # already (m, n_subset)
        p=p1,
        returnAB=True,
        robust=False,
        epsilon=0.001,
        computeXtX=False,
        stepsFISTA=3,
        stepsAS=50,
        randominit=False,
        numThreads=1,
    )
    run_time = time.time() - t0
    out = dict(
        subsetZ=Z,
        subsetA=A,
        subsetB=B,
        subsetsSampleIdxs=sample_idxs,
        runTime=run_time,
    )
    out_path = Path(outputs_dir) / f"{data_name}output{subset_idx + 1}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"  Subset {subset_idx + 1:02d}: {run_time:.1f} s  →  {out_path.name}", flush=True)
    return run_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    print(f"=== IAA on MNP  k={K}  p1={P1}  p2={P2}  seed={RANDOM_STATE} ===", flush=True)
    print(f"Run dir: {RUN_DIR}", flush=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading + standardising data...", flush=True)
    with open(DATA_PATH, "rb") as f:
        df = pickle.load(f)

    missing = df.isna().any(axis=1)
    if missing.any():
        print(f"  Dropping {int(missing.sum())} row(s) with NaN features", flush=True)
        df = df.loc[~missing]

    drop_cols = [c for c in LABEL_COLS if c in df.columns]
    feats = df.drop(columns=drop_cols).astype(np.float64)
    n, m = feats.shape
    print(f"  Data: n={n}  m={m}", flush=True)

    sc = StandardScaler()
    Xscaled = sc.fit_transform(feats.to_numpy())

    # Save scaler and kept-index mapping for the plotting script
    with open(RESULTS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(sc, f)
    with open(RESULTS_DIR / "kept_row_indices.pkl", "wb") as f:
        pickle.dump(df.index.tolist(), f)
    print("  Saved scaler.pkl and kept_row_indices.pkl", flush=True)

    # ------------------------------------------------------------------
    # Round 1: split + parallel subset AA
    # ------------------------------------------------------------------
    print(f"\n[2/4] Round 1: splitting into {K} subsets + running AA in parallel (p1={P1})...",
          flush=True)
    kf = KFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)
    subsets = [
        (i, idxs, Xscaled[idxs].T)   # X_subset_T is (m, n_subset) for SPAMS
        for i, (_, idxs) in enumerate(kf.split(Xscaled))
    ]

    all_done = all((OUTPUTS_DIR / f"{DATA_NAME}output{i + 1}.pkl").exists() for i in range(K))
    if all_done:
        print(f"  All {K} subset output pickles already exist in {OUTPUTS_DIR}. Skipping Round 1 AA.", flush=True)
        r1_wall = 0.0
    else:
        worker_args = [
            (i, idxs, X_sub_T, str(OUTPUTS_DIR), DATA_NAME, P1)
            for i, idxs, X_sub_T in subsets
        ]

        t_r1 = time.time()
        with ProcessPoolExecutor(max_workers=K) as ex:
            r1_times = list(ex.map(_subset_worker, worker_args))
        r1_wall = time.time() - t_r1
        print(f"  Round 1 done:  wall={r1_wall:.1f} s  max_subset={max(r1_times):.1f} s", flush=True)

    # ------------------------------------------------------------------
    # Round 2: final AA on k*p1=5000 intermediate archetypes
    # ------------------------------------------------------------------
    print(f"\n[3/4] Round 2: final AA on {K * P1} intermediates  p2={P2}  threads={N_THREADS_R2}...",
          flush=True)
    os.environ["OMP_NUM_THREADS"] = str(N_THREADS_R2)
    t_r2 = time.time()
    AA = fitPIAA(
        Xscaled,
        nArchetypes=P2,
        numSubset=K,
        dataName=DATA_NAME,
        outputsPicklesPath=str(OUTPUTS_DIR),
        shuffle=True,
        randomState=RANDOM_STATE,
        numThreads=N_THREADS_R2,
        splitRunTime=r1_wall,
        verbose=True,
    )
    r2_wall = time.time() - t_r2
    print(f"  Round 2 done:  wall={r2_wall:.1f} s", flush=True)

    # ------------------------------------------------------------------
    # Extract close matches + save
    # ------------------------------------------------------------------
    print("\n[4/4] Extracting close matches + saving...", flush=True)
    AA._extractCloseMatch()

    result_path = RESULTS_DIR / f"{DATA_NAME}_p2_{P2}_PIAA.pkl"
    with open(result_path, "wb") as f:
        pickle.dump(AA, f)

    total = time.time() - t_start
    print(f"\n=== DONE in {total:.1f} s  ({total / 3600:.2f} h) ===", flush=True)
    print(f"  Explained variance : {AA.explainedVariance_:.4f}", flush=True)
    print(f"  Result saved to    : {result_path}", flush=True)
    print(f"\nNext step: run plotFig4a_MNP_p1_500.py to generate Figure 4a panel.", flush=True)


if __name__ == "__main__":
    main()
