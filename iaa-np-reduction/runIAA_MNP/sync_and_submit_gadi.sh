#!/bin/bash
# Sync Round 1 subsets and scripts to Gadi and submit the Round 2 job.
set -euo pipefail

GADI_HOST="${1:-gadi.nci.org.au}"
GADI_USER="${2:-jt5911}"
REMOTE="${GADI_USER}@${GADI_HOST}"
REMOTE_REPO="/scratch/p00/${GADI_USER}/iterative-archetypal-analysis"

LOCAL_REPO="/home/DART/jonathant/Documents/Personal/PhD/iterative-archetypal-analysis"

echo "=== 1. Creating remote directory on Gadi ==="
ssh "${REMOTE}" "mkdir -p ${REMOTE_REPO}/iaa-np-reduction/runIAA_MNP/run_p1_500_gadi/subsetsOutputsPKLs"

echo "=== 2. Copying updated iteraa code (bugfix) ==="
scp "${LOCAL_REPO}/iteraa/src/iteraa/iaa.py" "${REMOTE}:${REMOTE_REPO}/iteraa/src/iteraa/iaa.py"

echo "=== 3. Copying Round 2 runner & PBS script ==="
scp "${LOCAL_REPO}/iaa-np-reduction/runIAA_MNP/runIAA_MNP_round2_gadi.py" "${REMOTE}:${REMOTE_REPO}/iaa-np-reduction/runIAA_MNP/"
scp "${LOCAL_REPO}/iaa-np-reduction/runIAA_MNP/run_round2_gadi.pbs" "${REMOTE}:${REMOTE_REPO}/iaa-np-reduction/runIAA_MNP/"

echo "=== 4. Copying 10 Round 1 subset pickles (~6.4 MB) ==="
rsync -avz --progress \
  "${LOCAL_REPO}/iaa-np-reduction/runIAA_MNP/local_run_p1_500/subsetsOutputsPKLs/" \
  "${REMOTE}:${REMOTE_REPO}/iaa-np-reduction/runIAA_MNP/run_p1_500_gadi/subsetsOutputsPKLs/"

echo "=== 5. Submitting job on Gadi ==="
ssh "${REMOTE}" "cd ${REMOTE_REPO} && qsub iaa-np-reduction/runIAA_MNP/run_round2_gadi.pbs"

echo "=== Done! The job has been submitted to Gadi. ==="
