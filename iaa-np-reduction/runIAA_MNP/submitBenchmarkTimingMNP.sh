#!/bin/bash
#
# Submit the MNP timing benchmark to a PBS Pro cluster (Gadi/NCI).
#
#   ./submitBenchmarkTimingMNP.sh --dry-run     # print the qsub command, submit nothing
#   ./submitBenchmarkTimingMNP.sh               # production configuration
#   ./submitBenchmarkTimingMNP.sh -k 10 -r 3 -W 10 -c 12 -w 04:00:00
#
# Every resource request and benchmark knob is a flag; nothing has to be edited.
# The job runs one process on one node: k subsets on WORKERS cores, then the
# final round(s) and the AA baseline. See runBenchmarkTimingMNP.sh for the
# measurement design, runtime expectations and the memory-bandwidth caveat.
#
# Multi-valued knobs use colons on this command line (--sweep 100:250:500)
# because PBS `-v` splits its argument on commas.
#
# Interpreter: the job needs Python >= 3.11 with spams. Pass -p
# /path/to/iteraa/.venv/bin/python, or set --module to a cluster module that
# provides spams. Without either, the job fails immediately with instructions
# rather than after loading the dataset.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PBS_SCRIPT="$SCRIPT_DIR/runBenchmarkTimingMNP.sh"

QUEUE=normal
PROJECT=p00
NAME=iaaTimingMNP
NCPUS=12
WALLTIME=24:00:00
MEM=16GB
JOBFS=1GB
EMAIL=jonathant@diversityarrays.com

K=10
P1=300
P2=250
SWEEP=100:250:500:1000
REPEAT=3
WORKERS=10
THREADS=1
RAA=1
RAA_THREADS=10
RAA_REPEATS=3
RAA_P2=""
LABEL=MNP
DATA=""
PYTHON=""
MODULE=""
OUTDIR="$SCRIPT_DIR/timingBenchmarkMNP_results"
RESUME=1
DRYRUN=0

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options (defaults in brackets):
  -n, --dry-run            print the qsub command and exit
  -q, --queue QUEUE        PBS queue                        [normal]
  -P, --project PROJECT    PBS project                      [p00]
  -N, --name NAME          job name                         [iaaTimingMNP]
  -c, --ncpus N            cores requested                  [12]
  -w, --walltime HH:MM:SS  walltime                         [24:00:00]
  -m, --mem SIZE           memory                           [16GB]
      --jobfs SIZE         job filesystem                   [1GB]
  -M, --email ADDRESS      notification address             [jonathant@diversityarrays.com]
      --no-email           suppress email notifications
  -k, --subsets K          number of subsets                [10]
      --p1 N               intermediate archetypes          [300]
      --p2 N               headline p2                      [250]
      --sweep A:B:C        p2 values (colon-separated)      [100:250:500:1000]
  -r, --repeat N           IAA repeats                      [3]
  -W, --workers N          cores used in round 1            [10]
  -T, --threads N          spams threads per AA call        [1]
      --raa-threads N      threads for the AA baseline      [10]
      --raa-repeats N      AA baseline repeats              [3]
      --raa-p2 A:B         p2 values for the AA baseline    [same as --sweep]
      --no-raa             skip the AA baseline entirely
      --resume             resume from partial checkpoint   [default]
      --no-resume          ignore checkpoint, run fresh
  -l, --label NAME         output file prefix               [MNP]
  -d, --data PATH          feature pickle                   [repo copy]
  -o, --outdir DIR         output directory                 [<script dir>/timingBenchmarkMNP_results]
  -p, --python PATH        interpreter with iteraa deps      [python3 from PATH]
      --module NAME        environment module to load        [none]
  -h, --help               this message
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

# Print a copy-pasteable command line: quote only arguments that need it, so that
# values such as `-l ncpus=48,walltime=12:00:00` stay readable.
printCmd() {
    printf '  %s' "$1"; shift
    local arg
    for arg in "$@"; do
        if [[ "$arg" == *[[:space:]\'\"]* || "$arg" == *\\* ]]; then
            printf " '%s'" "$arg"
        else
            printf ' %s' "$arg"
        fi
    done
    printf '\n'
}

while (($#)); do
    case "$1" in
        -n|--dry-run)      DRYRUN=1 ;;
        -q|--queue)        QUEUE=$2; shift ;;
        -P|--project)      PROJECT=$2; shift ;;
        -N|--name)         NAME=$2; shift ;;
        -c|--ncpus)        NCPUS=$2; shift ;;
        -w|--walltime)     WALLTIME=$2; shift ;;
        -m|--mem)          MEM=$2; shift ;;
        --jobfs)           JOBFS=$2; shift ;;
        -M|--email)        EMAIL=$2; shift ;;
        --no-email)        EMAIL="" ;;
        -k|--subsets)      K=$2; shift ;;
        --p1)              P1=$2; shift ;;
        --p2)              P2=$2; shift ;;
        --sweep)           SWEEP=$2; shift ;;
        -r|--repeat)       REPEAT=$2; shift ;;
        -W|--workers)      WORKERS=$2; shift ;;
        -T|--threads)      THREADS=$2; shift ;;
        --raa-threads)     RAA_THREADS=$2; shift ;;
        --raa-repeats)     RAA_REPEATS=$2; shift ;;
        --raa-p2)          RAA_P2=$2; shift ;;
        --no-raa)          RAA=0 ;;
        --resume)          RESUME=1 ;;
        --no-resume)       RESUME=0 ;;
        -l|--label)        LABEL=$2; shift ;;
        -d|--data)         DATA=$2; shift ;;
        -o|--outdir)       OUTDIR=$2; shift ;;
        -p|--python)       PYTHON=$2; shift ;;
        --module)          MODULE=$2; shift ;;
        -h|--help)         usage; exit 0 ;;
        *)                 die "unknown option '$1' (try --help)" ;;
    esac
    shift
done

[[ -r "$PBS_SCRIPT" ]] || die "job script not found: $PBS_SCRIPT"

# ---- validation -------------------------------------------------------------
for pair in "ncpus:$NCPUS" "subsets:$K" "p1:$P1" "p2:$P2" "repeat:$REPEAT" \
            "workers:$WORKERS" "threads:$THREADS" "raa-threads:$RAA_THREADS" \
            "raa-repeats:$RAA_REPEATS"; do
    name=${pair%%:*}; value=${pair#*:}
    [[ "$value" =~ ^[0-9]+$ ]] || die "--$name must be a non-negative integer (got '$value')"
done
(( NCPUS > 0 )) || die "--ncpus must be at least 1"
(( K > 0 )) || die "--subsets must be at least 1"
(( P1 > 0 )) || die "--p1 must be at least 1"
(( REPEAT > 0 )) || die "--repeat must be at least 1"
(( THREADS > 0 )) || die "--threads must be at least 1"
(( WORKERS > 0 )) || die "--workers must be at least 1"
(( WORKERS <= NCPUS )) || \
    die "--workers ($WORKERS) exceeds --ncpus ($NCPUS); raise -c or lower -W"

# p2 values must be < k*p1 (spams cannot return more archetypes than samples).
IFS=':' read -r -a sweepValues <<< "$SWEEP"
(( ${#sweepValues[@]} > 0 )) || die "--sweep is empty"
for p2 in "${sweepValues[@]}"; do
    [[ "$p2" =~ ^[0-9]+$ ]] || die "--sweep values must be integers (got '$p2')"
    (( p2 > 0 && p2 < K * P1 )) || die \
        "--sweep value $p2 must satisfy 0 < p2 < k*p1 = $((K * P1)); see comment R1.2"
done
if [[ -n "$RAA_P2" ]]; then
    IFS=':' read -r -a raaValues <<< "$RAA_P2"
    for p2 in "${raaValues[@]}"; do
        [[ "$p2" =~ ^[0-9]+$ ]] || die "--raa-p2 values must be integers (got '$p2')"
    done
fi

if [[ -n "$PYTHON" ]]; then
    [[ -x "$PYTHON" ]] || die "--python is not executable: $PYTHON"
    PYTHON=$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")
fi
if [[ -n "$DATA" && ! -r "$DATA" ]]; then
    die "--data is not readable: $DATA"
fi

mkdir -p "$OUTDIR" || die "cannot create output directory: $OUTDIR"
OUTDIR=$(cd "$OUTDIR" && pwd)
if [[ -n "$DATA" ]]; then
    DATA=$(cd "$(dirname "$DATA")" && pwd)/$(basename "$DATA")
fi

# ---- build the qsub command -------------------------------------------------
QSUB_ARGS=(
    -q "$QUEUE"
    -P "$PROJECT"
    -N "$NAME"
    -l "ncpus=$NCPUS,walltime=$WALLTIME,mem=$MEM,jobfs=$JOBFS"
)
if [[ -n "$EMAIL" ]]; then
    QSUB_ARGS+=(-M "$EMAIL")
else
    QSUB_ARGS+=(-m n)   # override the #PBS -m a default in the job script
fi

VARS="K=$K,P1=$P1,P2=$P2,P2_SWEEP=$SWEEP"
VARS+=",WORKERS=$WORKERS,THREADS=$THREADS,REPEAT=$REPEAT"
VARS+=",RAA=$RAA,RAA_THREADS=$RAA_THREADS,RAA_REPEATS=$RAA_REPEATS"
VARS+=",RESUME=$RESUME"
VARS+=",LABEL=$LABEL,OUTDIR=$OUTDIR"
[[ -n "$RAA_P2" ]] && VARS+=",RAA_P2=$RAA_P2"
[[ -n "$PYTHON" ]] && VARS+=",PYTHON=$PYTHON"
[[ -n "$MODULE" ]] && VARS+=",MODULE=$MODULE"
[[ -n "$DATA" ]]   && VARS+=",DATA=$DATA"
QSUB_ARGS+=(-v "$VARS")

echo "Benchmark : $LABEL  (k=$K, p1=$P1, p2=$SWEEP, repeats=$REPEAT)"
echo "Resources : ${NCPUS} cores, ${MEM} mem, walltime ${WALLTIME}, queue ${QUEUE}/${PROJECT}"
echo "Output    : $OUTDIR"
echo "Workers   : $WORKERS (round 1), threads per AA call: $THREADS"
echo "AA baseline: enabled=$RAA, threads=$RAA_THREADS, repeats=$RAA_REPEATS"
echo "Resume    : enabled=$RESUME"

if (( DRYRUN )); then
    echo
    echo "Dry run - would execute:"
    printCmd qsub "${QSUB_ARGS[@]}" "$PBS_SCRIPT"
    exit 0
fi

command -v qsub >/dev/null 2>&1 || die "qsub not found; run this on the cluster login node or use --dry-run"

JOBID=$(qsub "${QSUB_ARGS[@]}" "$PBS_SCRIPT")
JOBID=${JOBID%%[[:space:]]*}
echo
echo "Submitted job: $JOBID"

{
    echo "submitted $(date -u +%Y-%m-%dT%H:%M:%SZ) as $JOBID"
    printCmd qsub "${QSUB_ARGS[@]}" "$PBS_SCRIPT"
} > "$OUTDIR/submission_${JOBID}.txt"

cat <<EOF

Follow it with:
  qstat -u \$USER                    # queue status
  tail -f "$OUTDIR/${LABEL}_benchmark.out"
Once it finishes, the deliverables are:
  $OUTDIR/timingBenchmark_${LABEL}_table.tex     manuscript table
  $OUTDIR/timingBenchmark_${LABEL}_summary.csv   mean/std/min/max per quantity and p2
  $OUTDIR/timingBenchmark_${LABEL}_raw.json      per-repeat records + environment
EOF
