#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l ncpus=48,walltime=12:00:00,mem=32GB,jobfs=1GB
#PBS -l storage=scratch/q27
#PBS -l wd
#PBS -M Jonathan.Ting@anu.edu.au
#PBS -m a
#
# PBS job script: timing benchmark for IAA on the MNP dataset (comment R1.1).
# Measures core-seconds separately from wall-clock time for IAA (k, p1, p2) and
# for standard AA on the full dataset, and writes JSON/CSV/LaTeX results.
#
# Submit with the driver (recommended - it sets every resource and knob for you):
#   ./submitBenchmarkTimingMNP.sh --dry-run      # inspect the qsub command
#   ./submitBenchmarkTimingMNP.sh                # production configuration
#   ./submitBenchmarkTimingMNP.sh -k 10 -r 3 -W 10 -w 04:00:00 -c 12
#
# Or submit directly, overriding resources on the qsub command line (PBS Pro
# lets command-line -l/-q/-P/-N/-M attributes override the #PBS lines above):
#   qsub -l ncpus=12,walltime=04:00:00,mem=16GB runBenchmarkTimingMNP.sh
#   qsub -v P2_SWEEP=250:500:1000,REPEAT=5 runBenchmarkTimingMNP.sh
#
# NOTE ON LISTS: PBS `-v` splits on commas, so any multi-valued knob must be
# colon-separated here (P2_SWEEP=100:250:500). Values already colon-separated are
# converted to commas for the Python script below, which expects commas.
#
# Defaults are the production MNP configuration: k = 10 subsets, p1 = 300
# intermediate archetypes (runIAA_MNP_250As.out), swept over
# p2 = 100, 250, 500, 1000. Round 1 does not depend on p2, so all four points
# come from a single round-1 pass; only the final round and the AA baseline are
# repeated per p2. p2 = 250 is the published configuration and is the value used
# for the "Run i: x s" lines consumed by complexityScaling.ipynb.
#
# Rough runtime, measured on a 14-core workstation (n = 9299, m = 134, k = 10,
# p1 = 300):
#   round 1, serial (cost measurement)   270 s      (27 s per subset)
#   round 1, 10 workers (wall-clock)      65 s
#   final rounds, p2 = 100/250/500/1000   33/86/189/458 s
#   i.e. ~18 min per repeat, ~55 min for the default 3 repeats
#   AA baseline on the full dataset, one thread: 132/316/737/1877 s at
#   p2 = 100/250/500/1000, i.e. 51 min for the whole sweep. The same sweep with
#   RAA_THREADS=10 takes ~20 min: unlike the small subset calls, the full-dataset
#   AA scales 2.4-2.6x on 10 threads (737 s -> 305 s at p2 = 500, 1877 s ->
#   732 s at p2 = 1000).
#   Budget ~1.3 h in total with a threaded AA baseline, ~1.8 h with one thread,
#   ~2 h with RAA_REPEATS=3. Lower REPEAT (or drop the largest p2 from
#   P2_SWEEP) to shorten.
#
# Set RAA_THREADS to the same value as WORKERS when the AA baseline's
# wall-clock is compared with IAA's: the IAA row uses WORKERS cores, so an AA
# row measured on one core would not be a like-for-like comparison.
#
# Scaling note: only min(k, WORKERS) cores are used, so ncpus=48 is headroom for
# larger k, not a requirement. All subsets run in one process pool on one node,
# where they contend for memory bandwidth: at k = 10 each subset takes ~2.4x
# longer than it does alone (27 s -> 65 s), i.e. parallel efficiency ~0.41
# rather than 1.0. That contention is real for the production setup too, which
# submitted the subsets as concurrent PBS jobs; for the ideal Eq. (4) wall-clock,
# run the subsets as separate jobs on separate nodes.
#
# Interpreter: iteraa requires Python >= 3.11 and spams. Pass a virtual
# environment that has both, either through the driver (-p) or directly:
#   qsub -v PYTHON=/scratch/q27/jt5911/iterative-archetypal-analysis/iteraa/.venv/bin/python \
#        runBenchmarkTimingMNP.sh

set -euo pipefail

# Optional environment module. Empty by default: `module load python3/3.10.4`
# cannot satisfy iteraa's Python >= 3.11 requirement, so the venv interpreter is
# the reliable path. Set MODULE=python3/3.11.9 (or whatever this cluster
# provides) if it ships spams.
if [[ -n "${MODULE:-}" ]]; then
    if command -v module >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source /etc/profile.d/modules.sh 2>/dev/null || true
        module load "$MODULE"
    else
        echo "Warning: MODULE=$MODULE requested but no 'module' command is available" >&2
    fi
fi

PYTHON=${PYTHON:-python3}
K=${K:-10}
P1=${P1:-300}
P2=${P2:-250}                       # headline p2 (published configuration)
P2_SWEEP=${P2_SWEEP//:/,}           # accept colon lists from `qsub -v`
RAA_P2=${RAA_P2:-}
RAA_P2=${RAA_P2//:/,}
WORKERS=${WORKERS:-10}              # P in Eq. (4); keep <= ncpus requested
REPEAT=${REPEAT:-3}
THREADS=${THREADS:-1}               # spams threads per AA call
RAA=${RAA:-1}                       # 1 = also time standard AA on the full dataset
RAA_THREADS=${RAA_THREADS:-1}       # set to $WORKERS for the threaded AA wall-clock
RAA_REPEATS=${RAA_REPEATS:-1}
LABEL=${LABEL:-MNP}
OUTDIR=${OUTDIR:-timingBenchmarkMNP_results}
DATA=${DATA:-}

# Pin BLAS/OpenMP pools to the per-call thread count: one thread per subset by
# default (the layout Eq. (4) describes, P >= k), or THREADS for a control run.
# The Python module reads these with setdefault, so exported values win.
export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS

DATA_ARGS=()
[[ -n "$DATA" ]] && DATA_ARGS=(--data "$DATA")

if [[ "$RAA" == "1" ]]; then
    RAA_ARGS=(--raa --raaThreads "$RAA_THREADS" --raaRepeats "$RAA_REPEATS")
    [[ -n "$RAA_P2" ]] && RAA_ARGS+=(--raaP2 "$RAA_P2")
else
    RAA_ARGS=(--no-raa)
fi

cd "${PBS_O_WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"
mkdir -p "$OUTDIR"

# Fail fast in the queue slot rather than after loading the dataset.
if ! "$PYTHON" -c "import numpy, pandas, sklearn, spams" 2>/dev/null; then
    echo "ERROR: $PYTHON cannot import numpy/pandas/scikit-learn/spams." >&2
    echo "       iteraa needs Python >= 3.11 with spams installed. Pass a suitable" >&2
    echo "       interpreter with -v PYTHON=/path/to/iteraa/.venv/bin/python, or set" >&2
    echo "       MODULE=python3/3.11.x if that module provides spams." >&2
    exit 1
fi

NCPUS=${PBS_NCPUS:-$(getconf _NPROCESSORS_ONLN)}
if (( WORKERS > NCPUS )); then
    echo "Warning: WORKERS=$WORKERS exceeds the $NCPUS CPUs requested; round-1" >&2
    echo "         wall-clock will include queueing for cores." >&2
fi

echo "Host      : $(hostname)"
echo "Job id    : ${PBS_JOBID:-local}"
# `nproc` reports OMP_NUM_THREADS when it is set, which we pin above, so ask for
# the CPU count directly instead.
echo "CPUs      : $NCPUS available (requested ${PBS_NCPUS:-n/a} via PBS)"
echo "Started   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Python    : $("$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
echo "Config    : k=$K p1=$P1 p2=$P2 sweep=$P2_SWEEP workers=$WORKERS threads=$THREADS repeats=$REPEAT raa=$RAA raaThreads=$RAA_THREADS"

"$PYTHON" -u benchmarkTimingMNP.py \
    --label "$LABEL" \
    --outdir "$OUTDIR" \
    --k "$K" --p1 "$P1" --p2 "$P2" --p2Sweep "$P2_SWEEP" \
    --workers "$WORKERS" --threads "$THREADS" \
    --repeat "$REPEAT" \
    "${RAA_ARGS[@]}" "${DATA_ARGS[@]}" \
    | tee "${OUTDIR}/${LABEL}_benchmark.out"

echo "Finished  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
