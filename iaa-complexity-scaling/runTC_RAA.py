"""Time-complexity benchmark — Reference Archetypal Analysis (RAA).

Runs RAA on synthetic high-dimensional data for a range of sample sizes and
records wall-clock time for each run.  Output files are written to
``data/runTimes/computationalComplexity/`` and named::

    TC_RAA_{numSamples}.out

Each output file contains one line per run::

    Run {i}: {duration:.3f} s

These files are consumed by ``complexityScaling.ipynb`` via its
``getAllDurations`` helper.

Usage (local / interactive)
---------------------------
    python runTC_RAA.py

Usage (Gadi PBS job)
---------------------------------------------
    python runTC_RAA.py <numSamples> <outputFile>
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
SAMPLE_ARCH_RATIO: int = 10   # numArchetypes = numSamples // SAMPLE_ARCH_RATIO
RANDOM_STATE: int = 42

# SPAMS backend parameters (match manuscript values)
ROBUST: bool = False
TOLERANCE: float = 0.001
COMPUTE_XTX: bool = False
STEPS_FISTA: int = 3
STEPS_AS: int = 50
RANDOM_INIT: bool = False
NUM_THREADS: int = -1          # -1 → use all available cores

# Sample sizes to sweep in local/interactive mode
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


def run_raa(X: np.ndarray, num_archetypes: int) -> float:
    """Fit RAA and return wall-clock duration in seconds."""
    from iteraa import ArchetypalAnalysis

    aa = ArchetypalAnalysis(
        nArchetypes=num_archetypes,
        iterative=False,
        robust=ROBUST,
        onlyZ=True,
        tolerance=TOLERANCE,
        computeXtX=COMPUTE_XTX,
        stepsFISTA=STEPS_FISTA,
        stepsAS=STEPS_AS,
        randominit=RANDOM_INIT,
        numThreads=NUM_THREADS,
    )
    start = time()
    aa.fit(X)
    return time() - start


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def benchmark_one(num_samples: int, output_file: str) -> None:
    """Run NUM_ITERATION RAA fits for num_samples and write results."""
    num_archetypes = num_samples // SAMPLE_ARCH_RATIO
    X = gen_data(num_samples, NUM_FEATURES)

    makedirs(dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as fh:
        for i in range(NUM_ITERATION):
            duration = run_raa(X, num_archetypes)
            line = f"Run {i}: {duration:.3f} s"
            print(line)
            fh.write(line + "\n")
    print(f"Results written to {output_file}")


def main() -> None:
    if len(sys.argv) == 3:
        # Called from a PBS job script with explicit arguments
        num_samples = int(sys.argv[1])
        output_file = sys.argv[2]
        benchmark_one(num_samples, output_file)
    else:
        # Interactive / local sweep over all configured sample sizes
        for num_samples in TC_NUM_SAMPLES:
            output_file = f"{OUTPUT_DIR}/TC_RAA_{num_samples}.out"
            print(f"\n=== RAA  n={num_samples:5d} ===")
            benchmark_one(num_samples, output_file)


if __name__ == "__main__":
    main()
