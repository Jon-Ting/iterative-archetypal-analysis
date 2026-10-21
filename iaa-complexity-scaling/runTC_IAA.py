"""Time-complexity benchmark — Iterative Archetypal Analysis (IAA).

Runs IAA on synthetic high-dimensional data for a range of sample sizes and
records wall-clock time for each run.  Output files are written to
``data/runTimes/computationalComplexity/`` and named::

    TC_PIAA_{numSamples}.out

Each output file contains one line per run::

    Run {i}: {duration:.3f} s

These files are consumed by ``complexityScaling.ipynb`` via its
``getAllDurations`` helper (keyword ``TC_PIAA_``).

Usage (local / interactive)
---------------------------
    python runTC_IAA.py

Usage (Gadi PBS job)
---------------------------------------------
    python runTC_IAA.py <numSamples> <outputFile>

IAA Configuration Notes
-----------------------
- numSubsets  = SAMPLE_ARCH_RATIO  (= 10), so each subset has n/10 samples.
- numArchetypes (p1 = p2) = numSamples // SAMPLE_ARCH_RATIO.
- The ``iterative=True`` path inside ArchetypalAnalysis handles subset
  splitting and the two-round AA internally; runTime_ is measured as the total
  wall-clock time of both rounds (split + subset AA + final AA).
"""

import sys
from os import makedirs
from os.path import dirname
from time import time

import numpy as np
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_ITERATION: int = 30
NUM_FEATURES: int = 1000
SAMPLE_ARCH_RATIO: int = 10   # numArchetypes = numSubsets = numSamples // ratio
RANDOM_STATE: int = 42

ROBUST: bool = False
TOLERANCE: float = 0.001
COMPUTE_XTX: bool = False
STEPS_FISTA: int = 3
STEPS_AS: int = 50
RANDOM_INIT: bool = False
NUM_THREADS: int = -1

SHUFFLE: bool = True

TC_NUM_SAMPLES: tuple = (100, 200, 400, 800, 1600, 3200)

OUTPUT_DIR: str = "data/runTimes/computationalComplexity"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

np.random.seed(RANDOM_STATE)


def gen_data(n: int, m: int) -> np.ndarray:
    """Return a standardised random dataset of shape (n, m)."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.uniform(-10, 10, size=(n, m))
    sc = StandardScaler()
    return sc.fit_transform(X)


def run_iaa(X: np.ndarray, num_archetypes: int, num_subsets: int) -> float:
    """Fit IAA (iterative=True) and return wall-clock duration in seconds."""
    from iteraa import ArchetypalAnalysis

    aa = ArchetypalAnalysis(
        nArchetypes=num_archetypes,
        iterative=True,
        nSubsets=num_subsets,
        shuffle=SHUFFLE,
        robust=ROBUST,
        onlyZ=True,
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
    return time() - start


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def benchmark_one(num_samples: int, output_file: str) -> None:
    """Run NUM_ITERATION IAA fits for num_samples and write results."""
    num_archetypes = num_samples // SAMPLE_ARCH_RATIO
    num_subsets = SAMPLE_ARCH_RATIO   # = 10
    X = gen_data(num_samples, NUM_FEATURES)

    makedirs(dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as fh:
        for i in range(NUM_ITERATION):
            duration = run_iaa(X, num_archetypes, num_subsets)
            line = f"Run {i}: {duration:.3f} s"
            print(line)
            fh.write(line + "\n")
    print(f"Results written to {output_file}")


def main() -> None:
    if len(sys.argv) == 3:
        num_samples = int(sys.argv[1])
        output_file = sys.argv[2]
        benchmark_one(num_samples, output_file)
    else:
        for num_samples in TC_NUM_SAMPLES:
            output_file = f"{OUTPUT_DIR}/TC_PIAA_{num_samples}.out"
            print(f"\n=== IAA  n={num_samples:5d} ===")
            benchmark_one(num_samples, output_file)


if __name__ == "__main__":
    main()
