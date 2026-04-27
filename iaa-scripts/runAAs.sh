#!/bin/bash

# This script automates the submission of PBS jobs for archetypal analysis on data subsets.
# Usage: ./runAAs.sh <DATA_NAME> <RUNAA_ARG> <NCPUS> <WALLTIME> <MEM_GB>
#   DATA_NAME: Prefix for the .pkl subset files in SUBSETS_DIR
#   RUNAA_ARG: Additional argument passed to runAA.py (e.g., number of archetypes)
#   NCPUS: Number of CPUs to request per job
#   WALLTIME: Walltime limit for the job (e.g., 01:00:00)
#   MEM_GB: Memory in GB to request per job

# Specify data set
DATA_NAME=$1

# Directory paths (Note: These are hardcoded to a specific scratch environment)
OUTPUTS_DIR="/scratch/q27/jt5911/iterative-archetypal-analysis/iaa/docs/data/subsetsOutputsPKLs"
SUBSETS_DIR="/scratch/q27/jt5911/iterative-archetypal-analysis/iaa/docs/data/subsetsDataPKLs"
JOBSCRIPT_DIR="/scratch/q27/jt5911/iterative-archetypal-analysis/iaa/docs/jobScripts"
SCRIPT_PATH="/scratch/q27/jt5911/iterative-archetypal-analysis/iaa/src/iaa/runAA.py"

# Create directory for PBS scripts if it doesn't exist
mkdir -p $JOBSCRIPT_DIR

# Create a list of all subset files matching the data name prefix
cd $SUBSETS_DIR
SUBSET_FILES=($(ls $DATA_NAME*.pkl))

# Generate and submit a PBS script for each subset file
cd ../../
echo "Running archetypal analysis for all subsets..."
for subsetFile in "${SUBSET_FILES[@]}"; do
    jobScript="${JOBSCRIPT_DIR}/${subsetFile%.pkl}.pbs"
    
    # Create PBS script dynamically
    cat > "$jobScript" << EOL
#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l ncpus=$3,walltime=$4,mem=$5GB,jobfs=1GB
#PBS -l storage=scratch/q27
#PBS -l wd
#PBS -M Jonathan.Ting@anu.edu.au
#PBS -m a

module load python3/3.10.4

cd \$PBS_O_WORKDIR
# Execute the archetypal analysis script
python3 $SCRIPT_PATH $SUBSETS_DIR/$subsetFile $2 $OUTPUTS_DIR

EOL

    # Submit the generated PBS script to the scheduler
    qsub "$jobScript"
    
    echo "  Submitted job for $subsetFile"
done

# echo -e "\nAll jobs submitted!"
