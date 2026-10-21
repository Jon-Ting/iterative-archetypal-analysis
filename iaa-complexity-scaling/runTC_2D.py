"""Time-complexity benchmark for manuscript Table 1 — 2D synthetic datasets.

Reproduces the benchmarks reported in Table 1 of the manuscript
(``tab:redRed2DdataEval``): RAA vs IAA on 5000-sample 2D square and circular
datasets, with IAA tested at 2, 4, and 8 subsets.  Each configuration is
repeated ``NUM_ITERATION`` times and the mean execution time and explained
variance are printed in a table-ready format.

Output files are written to ``data/runTimes/computationalComplexity/`` and are
named::

    TC_2D_RAA_square.out
    TC_2D_IAA_square_k{k}.out
    TC_2D_RAA_circular.out
    TC_2D_IAA_circular_k{k}.out

Each file contains one line per run::

    Run {i}: {duration:.3f} s  EV: {explained_variance:.4f}

Usage
-----
    python runTC_2D.py
"""

from os import makedirs
from time import time

import numpy as np
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_ITERATION: int = 30
RANDOM_STATE: int = 42

NUM_SAMPLES: int = 5000

# Number of archetypes per dataset (match the manuscript)
ARCHETYPES_SQUARE: int = 4
ARCHETYPES_CIRCULAR: int = 8

# Number of IAA subsets to test
IAA_SUBSETS: tuple = (2, 4, 8)

# SPAMS backend parameters (match the manuscript)
ROBUST: bool = False
TOLERANCE: float = 0.001
COMPUTE_XTX: bool = False
STEPS_FISTA: int = 3
STEPS_AS: int = 50
RANDOM_INIT: bool = False
NUM_THREADS: int = -1
SHUFFLE: bool = True

OUTPUT_DIR: str = "data/runTimes/computationalComplexity"
makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

np.random.seed(RANDOM_STATE)


def gen_square(n: int) -> np.ndarray:
    """Uniform square dataset, standardised. Shape: (n, 2)."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.uniform(-10, 10, size=(n, 2))
    sc = StandardScaler()
    return sc.fit_transform(X)


def gen_circular(n: int) -> np.ndarray:
    """Uniform disk dataset, standardised. Shape: (n, 2)."""
    rng = np.random.default_rng(RANDOM_STATE)
    # Rejection-sample within unit circle, then scale
    samples = []
    while len(samples) < n:
        pts = rng.uniform(-1, 1, size=(n * 2, 2))
        inside = pts[np.sum(pts ** 2, axis=1) <= 1.0]
        samples.extend(inside.tolist())
    X = np.array(samples[:n]) * 10.0
    sc = StandardScaler()
    return sc.fit_transform(X)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_raa(X: np.ndarray, num_archetypes: int) -> tuple:
    """Run RAA, return (duration_s, explained_variance)."""
    from iteraa import ArchetypalAnalysis

    aa = ArchetypalAnalysis(
        nArchetypes=num_archetypes,
        iterative=False,
        robust=ROBUST,
        onlyZ=False,   # Need alfa for explained variance
        tolerance=TOLERANCE,
        computeXtX=COMPUTE_XTX,
        stepsFISTA=STEPS_FISTA,
        stepsAS=STEPS_AS,
        randominit=RANDOM_INIT,
        numThreads=NUM_THREADS,
    )
    start = time()
    aa.fit(X)
    duration = time() - start
    ev = aa.explainedVariance_ if aa.explainedVariance_ is not None else float("nan")
    return duration, ev


def run_iaa(X: np.ndarray, num_archetypes: int, num_subsets: int) -> tuple:
    """Run IAA, return (duration_s, explained_variance)."""
    from iteraa import ArchetypalAnalysis

    aa = ArchetypalAnalysis(
        nArchetypes=num_archetypes,
        iterative=True,
        nSubsets=num_subsets,
        shuffle=SHUFFLE,
        robust=ROBUST,
        onlyZ=False,
        tolerance=TOLERANCE,
        computeXtX=COMPUTE_XTX,
        stepsFISTA=STEPS_FISTA,
        stepsAS=STEPS_AS,
        randominit=RANDOM_INIT,
        randomState=RANDOM_STATE,
        numThreads=NUM_THREADS,
    )
    start = time()
    aa.fit(X)
    duration = time() - start
    ev = aa.explainedVariance_ if aa.explainedVariance_ is not None else float("nan")
    return duration, ev


# ---------------------------------------------------------------------------
# Benchmark one configuration
# ---------------------------------------------------------------------------

def benchmark(label: str, X: np.ndarray, num_archetypes: int,
              runner, output_file: str, **runner_kwargs) -> dict:
    """Run NUM_ITERATION times, write output file, return summary stats."""
    durations = []
    evs = []
    with open(output_file, "w") as fh:
        for i in range(NUM_ITERATION):
            duration, ev = runner(X, num_archetypes, **runner_kwargs)
            line = f"Run {i}: {duration:.3f} s  EV: {ev:.4f}"
            print(line)
            fh.write(line + "\n")
            durations.append(duration)
            evs.append(ev)
    mean_dur = np.mean(durations)
    mean_ev = np.mean(evs) * 100.0
    print(f"  → mean duration: {mean_dur:.1f} s   mean EV: {mean_ev:.1f}%")
    return {"label": label, "duration": mean_dur, "ev": mean_ev}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    datasets = [
        ("square", gen_square(NUM_SAMPLES), ARCHETYPES_SQUARE),
        ("circular", gen_circular(NUM_SAMPLES), ARCHETYPES_CIRCULAR),
    ]

    results = []
    for name, X, n_arch in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {name}  (n={NUM_SAMPLES}, k={n_arch})")
        print(f"{'='*60}")

        # --- RAA ---
        print("\nRAA:")
        out = f"{OUTPUT_DIR}/TC_2D_RAA_{name}.out"
        res = benchmark(f"RAA ({name})", X, n_arch, run_raa, out)
        results.append(res)

        # --- IAA for each subset count ---
        for k in IAA_SUBSETS:
            print(f"\nIAA  k={k}:")
            out = f"{OUTPUT_DIR}/TC_2D_IAA_{name}_k{k}.out"
            res = benchmark(f"IAA k={k} ({name})", X, n_arch, run_iaa, out,
                            num_subsets=k)
            results.append(res)

    # Print Table-1 style summary
    print(f"\n{'='*60}")
    print("Summary (manuscript Table 1 style)")
    print(f"{'='*60}")
    print(f"{'Label':<30} {'Duration (s)':>14} {'Expl. Var (%)':>14}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:<30} {r['duration']:>14.1f} {r['ev']:>14.1f}")


if __name__ == "__main__":
    main()
