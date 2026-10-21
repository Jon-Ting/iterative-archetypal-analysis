"""Weak-scaling benchmark for IAA.

Weak scaling increases both the number of parallel processors and the problem
size proportionally, keeping the workload per processor constant.  Ideal
weak-scaling efficiency is 1.0 (constant duration regardless of scale).

The test case matches the notebook (Gadi, ``complexityScaling.ipynb``, §Weak
Scaling).  In the original run, the data came from molecular-dynamics
nanoparticle simulations (voxelised-surface descriptor, hence the "VX" label),
where atom counts were used to set problem sizes.  These scripts reproduce the
same experiment with synthetic data using the same atom-count values as sample
counts, so the relative scaling behaviour is preserved.

Configuration
-------------
WS_Ncpus    = (1, 2, 4, 8, 16, 24, 48, 96)
WS_Nnodes   = (1, 1, 1, 1,  1,  1,  1,  2)
WS_NumAtoms = (43, 87, 177, 321, 683, 1055, 1961, 4189)  # actual atom counts
               → used directly as n_samples for each configuration

Each entry in WS_Ncpus is the total number of CPUs; the number of subsets
equals the number of CPU sockets used per node (inferred from the Gadi spec:
2 sockets per node, 24 cores per socket).

Output files
------------
``runtimes/weakScaling/WS_VX_{ncpus}.out``

Each file contains NUM_ITERATION lines::

    Run {i}: {duration:.3f} s

Wall-clock is estimated as one_subset_time + final_AA_time (same approach as
``runSS.py``; see that script for a detailed explanation).

Usage
-----
Local (sequential simulation):
    python runWS.py

PBS (called from submitWS.sh for one CPU-count step):
    python runWS.py <ncpus> <nSubsets> <nSamples> <numThreads> <outputFile>
"""

import sys
from os import makedirs
from os.path import dirname
from time import time

import numpy as np
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration  (from complexityScaling.ipynb cells 63-64)
# ---------------------------------------------------------------------------
NUM_ITERATION: int = 30
NUM_FEATURES: int = 1000
SAMPLE_ARCH_RATIO: int = 10      # numArchetypes = nSamples // ratio
RANDOM_STATE: int = 42

ROBUST: bool = False
TOLERANCE: float = 0.001
COMPUTE_XTX: bool = False
STEPS_FISTA: int = 3
STEPS_AS: int = 50
RANDOM_INIT: bool = False
SHUFFLE: bool = True

# Gadi Cascade Lake: 2 NUMA nodes per node, 12 cores per NUMA node → 24 per socket
# → nSubsets = ncpus // 24 for intra-node; 1 subset per node for multi-node
WS_NCPUS: tuple = (1,  2,   4,   8,   16,   24,   48,   96)
WS_NNODES: tuple = (1,  1,   1,   1,    1,    1,    1,    2)
WS_NSUBSETS: tuple = (1, 1,   1,   1,    1,    1,    2,    4)  # subsets per run
WS_NUM_ATOMS: tuple = (43, 87, 177, 321, 683, 1055, 1961, 4189)
# Threads per subset = ncpus / nSubsets
# e.g. 48 cpus, 2 subsets → 24 threads each

OUTPUT_DIR: str = "runtimes/weakScaling"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

np.random.seed(RANDOM_STATE)


def gen_data(n: int, m: int) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.uniform(-10, 10, size=(n, m))
    sc = StandardScaler()
    return sc.fit_transform(X)


def time_one_subset(X: np.ndarray, num_archetypes: int,
                    n_subsets: int, num_threads: int) -> float:
    """Time a single subset AA (represents the parallel wall-clock contribution)."""
    from spams import archetypalAnalysis

    rng = np.random.default_rng(RANDOM_STATE)
    n = X.shape[0]
    subset_size = n // n_subsets
    idxs = rng.choice(n, subset_size, replace=False)
    subsetX = X[idxs, :].T

    start = time()
    archetypalAnalysis(
        np.asfortranarray(subsetX),
        Z0=None,
        p=num_archetypes,
        returnAB=False,
        robust=ROBUST,
        epsilon=TOLERANCE,
        computeXtX=COMPUTE_XTX,
        stepsFISTA=STEPS_FISTA,
        stepsAS=STEPS_AS,
        randominit=RANDOM_INIT,
        numThreads=num_threads,
    )
    return time() - start


def time_final_aa(n_subsets: int, num_archetypes: int,
                  num_threads: int) -> float:
    """Time the final AA on pooled intermediate archetypes."""
    from spams import archetypalAnalysis

    rng = np.random.default_rng(RANDOM_STATE)
    intermediate = rng.standard_normal(
        (NUM_FEATURES, n_subsets * num_archetypes)
    )
    start = time()
    archetypalAnalysis(
        np.asfortranarray(intermediate),
        Z0=None,
        p=num_archetypes,
        returnAB=False,
        robust=ROBUST,
        epsilon=TOLERANCE,
        computeXtX=COMPUTE_XTX,
        stepsFISTA=STEPS_FISTA,
        stepsAS=STEPS_AS,
        randominit=RANDOM_INIT,
        numThreads=num_threads,
    )
    return time() - start


def estimate_wall_clock(X: np.ndarray, n_subsets: int,
                        num_archetypes: int, num_threads: int) -> float:
    """wall_clock ≈ one_subset_time + final_AA_time (parallel model)."""
    t_subset = time_one_subset(X, num_archetypes, n_subsets, num_threads)
    t_final = time_final_aa(n_subsets, num_archetypes, num_threads)
    return t_subset + t_final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def benchmark_one(ncpus: int, n_subsets: int, n_samples: int,
                  num_threads: int, output_file: str) -> None:
    """Run NUM_ITERATION wall-clock estimates and write to output_file."""
    num_archetypes = max(1, n_samples // SAMPLE_ARCH_RATIO)
    X = gen_data(n_samples, NUM_FEATURES)

    makedirs(dirname(output_file) or ".", exist_ok=True)
    print(f"  ncpus={ncpus}, n_subsets={n_subsets}, "
          f"n_samples={n_samples}, num_archetypes={num_archetypes}, "
          f"num_threads={num_threads}")
    with open(output_file, "w") as fh:
        for i in range(NUM_ITERATION):
            duration = estimate_wall_clock(X, n_subsets, num_archetypes,
                                           num_threads)
            line = f"Run {i}: {duration:.3f} s"
            print(line)
            fh.write(line + "\n")
    print(f"  Results written to {output_file}")


def main() -> None:
    if len(sys.argv) == 6:
        # PBS call: python runWS.py <ncpus> <nSubsets> <nSamples> <numThreads> <outputFile>
        ncpus = int(sys.argv[1])
        n_subsets = int(sys.argv[2])
        n_samples = int(sys.argv[3])
        num_threads = int(sys.argv[4])
        output_file = sys.argv[5]
        benchmark_one(ncpus, n_subsets, n_samples, num_threads, output_file)
    else:
        # Local sweep over all configurations
        for ncpus, n_subsets, n_samples in zip(
            WS_NCPUS, WS_NSUBSETS, WS_NUM_ATOMS
        ):
            num_threads = ncpus // max(n_subsets, 1)
            output_file = f"{OUTPUT_DIR}/WS_VX_{ncpus}.out"
            print(f"\n=== Weak Scaling  ncpus={ncpus}  "
                  f"n_subsets={n_subsets}  n_samples={n_samples} ===")
            benchmark_one(ncpus, n_subsets, n_samples, num_threads,
                          output_file)


if __name__ == "__main__":
    main()
