#!/bin/bash
# submitTC.sh — Submit time-complexity benchmark jobs to Gadi (PBS).
#
# Usage:
#   ./submitTC.sh          # submits all RAA and IAA jobs
#   ./submitTC.sh raa      # submits only RAA jobs
#   ./submitTC.sh iaa      # submits only IAA jobs
#   ./submitTC.sh 2d       # submits the 2D dataset benchmark (Table 1)
#
# Each job runs one (script, numSamples) pair and writes results to
#   data/runTimes/computationalComplexity/TC_{RAA|PIAA}_{numSamples}.out
#
# Adjust PROJECT, WALLTIME, NCPUS, and MEM as appropriate for your allocation.

PROJECT="p00"
QUEUE="normal"
NCPUS=48
WALLTIME_SHORT="04:00:00"    # for small n
WALLTIME_LONG="12:00:00"     # for n >= 1600
MEM=120                       # GB
EMAIL="jonathant@diversityarrays.com"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/data/runTimes/computationalComplexity"
JOBSCRIPTS_DIR="${SCRIPT_DIR}/jobScripts"

# Interpreter used inside the PBS jobs. The venv is NOT active in a batch job
# (PBS starts a clean non-interactive shell), so point at the venv directly
# instead of relying on `python3` from PATH, which is the bare module Python.
# Override with e.g. PYTHON=../iteraa/.venv/bin/python ./submitTC.sh
PYTHON="${PYTHON:-${SCRIPT_DIR}/.venv/bin/python}"

mkdir -p "${OUTPUT_DIR}" "${JOBSCRIPTS_DIR}"

# Fail at submit time rather than after every job has queued and died.
if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: no executable interpreter at ${PYTHON}" >&2
    echo "Create the venv first:" >&2
    echo "  cd ${SCRIPT_DIR}" >&2
    echo "  module load python3/3.11.7" >&2
    echo "  python3 -m venv .venv" >&2
    echo "  .venv/bin/pip install -e ../iteraa && .venv/bin/pip install -e ." >&2
    exit 1
fi
if ! "${PYTHON}" -c 'import numpy, scipy, sklearn, iteraa' >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON} cannot import numpy/scipy/sklearn/iteraa." >&2
    echo "Install them into that venv:" >&2
    echo "  ${PYTHON} -m pip install -e ${SCRIPT_DIR}/../iteraa" >&2
    echo "  ${PYTHON} -m pip install -e ${SCRIPT_DIR}" >&2
    exit 1
fi
echo "Interpreter: ${PYTHON} ($("${PYTHON}" -c 'import sys; print(sys.version.split()[0])'))"

TC_SAMPLES=(100 200 400 800 1600 3200)

# Decide walltime based on sample count
walltime_for_n() {
    local n=$1
    if [ "$n" -ge 1600 ]; then
        echo "${WALLTIME_LONG}"
    else
        echo "${WALLTIME_SHORT}"
    fi
}

submit_job() {
    local job_name=$1
    local script=$2
    local n=$3
    local out_file=$4
    local walltime=$(walltime_for_n $n)

    local pbs_script="${JOBSCRIPTS_DIR}/${job_name}_${n}.pbs"
    # Skip if output file already exists
    if [ -f "${out_file}" ]; then
        echo "  Skipping ${job_name} n=${n} — output already exists at ${out_file}"
        return 0
    fi
    cat > "${pbs_script}" << PBS
#!/bin/bash
#PBS -N ${job_name}_${n}
#PBS -P ${PROJECT}
#PBS -q ${QUEUE}
#PBS -l ncpus=${NCPUS},walltime=${walltime},mem=${MEM}GB,jobfs=1GB
#PBS -l storage=scratch/${PROJECT}
#PBS -l wd
#PBS -M ${EMAIL}
#PBS -m abe

module load python3/3.11.7

cd ${SCRIPT_DIR}
"${PYTHON}" -u ${script} ${n} ${out_file}
PBS

    qsub "${pbs_script}"
    echo "  Submitted ${job_name} n=${n} → ${out_file}"
}

MODE="${1:-all}"

# Submit RAA jobs
if [[ "$MODE" == "all" || "$MODE" == "raa" ]]; then
    echo "=== Submitting RAA jobs ==="
    for n in "${TC_SAMPLES[@]}"; do
        submit_job "TC_RAA" "runTC_RAA.py" "$n" \
            "${OUTPUT_DIR}/TC_RAA_${n}.out"
    done
fi

# Submit IAA (PIAA) jobs
if [[ "$MODE" == "all" || "$MODE" == "iaa" ]]; then
    echo "=== Submitting IAA jobs ==="
    for n in "${TC_SAMPLES[@]}"; do
        submit_job "TC_PIAA" "runTC_IAA.py" "$n" \
            "${OUTPUT_DIR}/TC_PIAA_${n}.out"
    done
fi

# Submit the 2D benchmark (Table 1)
if [[ "$MODE" == "all" || "$MODE" == "2d" ]]; then
    echo "=== Submitting 2D benchmark ==="
    pbs_script="${JOBSCRIPTS_DIR}/TC_2D.pbs"
    cat > "${pbs_script}" << PBS
#!/bin/bash
#PBS -N TC_2D
#PBS -P ${PROJECT}
#PBS -q ${QUEUE}
#PBS -l ncpus=${NCPUS},walltime=${WALLTIME_SHORT},mem=${MEM}GB,jobfs=1GB
#PBS -l storage=scratch/${PROJECT}
#PBS -l wd
#PBS -M ${EMAIL}
#PBS -m abe

module load python3/3.11.7

cd ${SCRIPT_DIR}
"${PYTHON}" -u runTC_2D.py
PBS
    qsub "${pbs_script}"
    echo "  Submitted 2D benchmark"
fi

echo "Done."

# ── Strong Scaling submission ──────────────────────────────────────────────
submit_ss_job() {
    local n_nodes=$1
    local total_cpus=$(( n_nodes * NCPUS ))

    local pbs_script="${JOBSCRIPTS_DIR}/SS_VX_${total_cpus}.pbs"
    cat > "${pbs_script}" << PBS
#!/bin/bash
#PBS -N SS_VX_${total_cpus}
#PBS -P ${PROJECT}
#PBS -q ${QUEUE}
#PBS -l ncpus=${total_cpus},walltime=${WALLTIME_SHORT},mem=$(( n_nodes * MEM ))GB,jobfs=1GB
#PBS -l storage=scratch/${PROJECT}
#PBS -l wd
#PBS -M ${EMAIL}
#PBS -m abe

module load python3/3.11.7

cd ${SCRIPT_DIR}
"${PYTHON}" -u runSS.py ${n_nodes} ${NCPUS} ${OUTPUT_DIR}/SS_VX_${total_cpus}.out
PBS

    qsub "${pbs_script}"
    echo "  Submitted SS n_nodes=${n_nodes} total_cpus=${total_cpus}"
}

if [[ "$MODE" == "all" || "$MODE" == "ss" ]]; then
    SS_OUTPUT_DIR="${SCRIPT_DIR}/data/runTimes/strongScaling"
    mkdir -p "${SS_OUTPUT_DIR}"
    echo "=== Submitting Strong Scaling jobs ==="
    for n_nodes in 1 2 4 8 16 32; do
        submit_ss_job $n_nodes
    done
fi

# ── Weak Scaling submission ────────────────────────────────────────────────
WS_NCPUS=(1 2 4 8 16 24 48 96)
WS_NSUBSETS=(1 1 1 1 1 1 2 4)
WS_NSAMPLES=(43 87 177 321 683 1055 1961 4189)
WS_NNODES=(1 1 1 1 1 1 1 2)

if [[ "$MODE" == "all" || "$MODE" == "ws" ]]; then
    WS_OUTPUT_DIR="${SCRIPT_DIR}/data/runTimes/weakScaling"
    mkdir -p "${WS_OUTPUT_DIR}"
    echo "=== Submitting Weak Scaling jobs ==="
    for idx in "${!WS_NCPUS[@]}"; do
        ncpus=${WS_NCPUS[$idx]}
        n_subsets=${WS_NSUBSETS[$idx]}
        n_samples=${WS_NSAMPLES[$idx]}
        n_nodes=${WS_NNODES[$idx]}
        num_threads=$(( ncpus / n_subsets ))

        pbs_script="${JOBSCRIPTS_DIR}/WS_VX_${ncpus}.pbs"
        cat > "${pbs_script}" << PBS
#!/bin/bash
#PBS -N WS_VX_${ncpus}
#PBS -P ${PROJECT}
#PBS -q ${QUEUE}
#PBS -l ncpus=${ncpus},walltime=${WALLTIME_SHORT},mem=$(( n_nodes * MEM ))GB,jobfs=1GB
#PBS -l storage=scratch/${PROJECT}
#PBS -l wd
#PBS -M ${EMAIL}
#PBS -m abe

module load python3/3.11.7

cd ${SCRIPT_DIR}
"${PYTHON}" -u runWS.py ${ncpus} ${n_subsets} ${n_samples} ${num_threads} ${WS_OUTPUT_DIR}/WS_VX_${ncpus}.out
PBS
        qsub "${pbs_script}"
        echo "  Submitted WS ncpus=${ncpus} n_subsets=${n_subsets} n_samples=${n_samples}"
    done
fi
