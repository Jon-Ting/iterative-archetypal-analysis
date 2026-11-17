#!/bin/bash
# Fetch Round 2 results from Gadi and run the Figure 4a plotting script.
set -euo pipefail

GADI_HOST="${1:-gadi.nci.org.au}"
GADI_USER="${2:-jt5911}"
REMOTE="${GADI_USER}@${GADI_HOST}"

# Paths are derived from this script's location, so the same script works from any
# checkout. Override with LOCAL_REPO=... / REMOTE_REPO=... if needed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="${LOCAL_REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
REMOTE_REPO="${REMOTE_REPO:-/scratch/p00/${GADI_USER}/iterative-archetypal-analysis}"
LOCAL_RESULTS="${LOCAL_REPO}/iaa-np-reduction/runIAA_MNP/local_run_p1_500/results"

mkdir -p "${LOCAL_RESULTS}"

echo "=== Fetching Round 2 results from Gadi... ==="
scp "${REMOTE}:${REMOTE_REPO}/iaa-np-reduction/runIAA_MNP/run_p1_500_gadi/results/*" "${LOCAL_RESULTS}/"

echo "=== Generating Figure 4a simplex plot locally... ==="
"${LOCAL_REPO}/iteraa/.venv/bin/python" "${LOCAL_REPO}/iaa-np-reduction/runIAA_MNP/plotFig4a_MNP_p1_500.py"

echo "=== Success! Check the generated figures in iaa-np-reduction/runIAA_MNP/local_run_p1_500/figs/ ==="
