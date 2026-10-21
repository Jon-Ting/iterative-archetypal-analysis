"""Strong-scaling benchmark for IAA.

Strong scaling fixes the total problem size and increases the number of
parallel processors (PBS nodes), measuring the reduction in wall-clock time.

In IAA, parallelism comes from distributing data subsets across PBS nodes.
Each node runs one subset using all its CPUs for SPAMS multi-threading.
Wall-clock time for one IAA run = time for one subset (all subsets run in
parallel) + time for the final AA round.

The test case matches the notebook (Gadi, ``complexityScaling.ipynb``, §Strong
Scaling): 6400 samples, 1000 features, 10% archetypes.

Configuration
-------------
SS_Ncpus  = (48, 96, 192, 384, 768, 1536)   # total CPUs
SS_Nnodes = ( 1,  2,   4,   8,  16,   32)   # PBS nodes; nSubsets = nNodes

Output files
------------
``runtimes/strongScaling/SS_VX_{ncpus}.out``

Each file contains NUM_ITERATION lines::

    Run {i}: {duration:.3f} s

Wall-clock duration is estimated as::

    t_wall ≈ t_one_subset + t_final_AA

because all subsets run in parallel on separate nodes.

Usage
-----
Local (simulates the timing, sequential):
    python runSS.py

PBS (called from submitSS.sh for a specific node count):
    python runSS.py <nNodes> <ncpus_per_node> <outputFile>

    nNodes          : number of parallel nodes (= number of subsets)
    ncpus_per_node  : CPUs per node (48 on Gadi normal queue)
    outputFile      : path to the output .out file
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
NUM_SAMPLES: int = 6400          # Fixed problem size (largest TC case)
NUM_FEATURES: int = 1000
SAMPLE_ARCH_RATIO: int = 10      # numArchetypes = NUM_SAMPLES // ratio = 640
RANDOM_STATE: int = 42

ROBUST: bool = False
TOLERANCE: float = 0.001
COMPUTE_XTX: bool = False
STEPS_FISTA: int = 3
STEPS_AS: int = 50
RANDOM_INIT: bool = False
SHUFFLE: bool = True

# Gadi normal-queue nodes → total CPUs mapping
SS_NCPUS_PER_NODE: int = 48
SS_NNODES: tuple = (1, 2, 4, 8, 16, 32)
# Derived: SS_Ncpus = tuple(n * SS_NCPUS_PER_NODE for n in SS_NNODES)

OUTPUT_DIR: str = "runtimes/strongScaling"

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
    """Time a single subset AA run (represents parallel wall-clock time).

    On HPC, all subsets run simultaneously, so the effective subset
    wall-clock time is determined by the slowest subset.  On balanced
    synthetic data the subsets are identical in size, so we time just one.
    """
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


def time_final_aa(intermediate_samples: np.ndarray,
                  num_archetypes: int, num_threads: int) -> float:
    """Time the final AA round on the pooled intermediate archetypes."""
    from spams import archetypalAnalysis

    start = time()
    archetypalAnalysis(
        np.asfortranarray(intermediate_samples),
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
    """Estimate wall-clock time = one_subset_time + final_aa_time.

    This mirrors what happens on HPC when all subsets run in parallel:
    wall_clock ≈ max(subset_times) + T(AA(k*p1, m, p2))
    With balanced data max ≈ mean ≈ one subset time.
    """
    t_subset = time_one_subset(X, num_archetypes, n_subsets, num_threads)

    # Build a dummy intermediate-archetypes matrix of the correct size:
    # k subsets × p1 archetypes each, same dimensionality as X
    rng = np.random.default_rng(RANDOM_STATE)
    intermediate = rng.standard_normal((NUM_FEATURES, n_subsets * num_archetypes))
    t_final = time_final_aa(intermediate, num_archetypes, num_threads)

    return t_subset + t_final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def benchmark_one(n_nodes: int, ncpus_per_node: int,
                  output_file: str) -> None:
    """Run NUM_ITERATION wall-clock estimates for n_nodes and write results."""
    num_archetypes = NUM_SAMPLES // SAMPLE_ARCH_RATIO
    n_subsets = n_nodes          # one subset per node
    num_threads = ncpus_per_node  # SPAMS uses all CPUs on the node
    total_cpus = n_nodes * ncpus_per_node

    X = gen_data(NUM_SAMPLES, NUM_FEATURES)
    makedirs(dirname(output_file) or ".", exist_ok=True)

    print(f"  n_nodes={n_nodes}, n_subsets={n_subsets}, "
          f"num_threads={num_threads}, total_cpus={total_cpus}")
    with open(output_file, "w") as fh:
        for i in range(NUM_ITERATION):
            duration = estimate_wall_clock(X, n_subsets, num_archetypes,
                                           num_threads)
            line = f"Run {i}: {duration:.3f} s"
            print(line)
            fh.write(line + "\n")
    print(f"  Results written to {output_file}")


def main() -> None:
    if len(sys.argv) == 4:
        # PBS call: python runSS.py <nNodes> <ncpus_per_node> <outputFile>
        n_nodes = int(sys.argv[1])
        ncpus_per_node = int(sys.argv[2])
        output_file = sys.argv[3]
        benchmark_one(n_nodes, ncpus_per_node, output_file)
    else:
        # Local sweep over all node counts
        for n_nodes in SS_NNODES:
            total_cpus = n_nodes * SS_NCPUS_PER_NODE
            output_file = f"{OUTPUT_DIR}/SS_VX_{total_cpus}.out"
            print(f"\n=== Strong Scaling  nodes={n_nodes}  cpus={total_cpus} ===")
            benchmark_one(n_nodes, SS_NCPUS_PER_NODE, output_file)


if __name__ == "__main__":
    main()
