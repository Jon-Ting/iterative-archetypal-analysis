#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l walltime=05:00:00,ncpus=96,mem=384GB,jobfs=100GB
#PBS -l storage=gdata/dk92+scratch/q27+gdata/q27
##PBS -v PYTHONPATH=<path to where you installed your Python packages>

# Script to generate core-shell bimetallic nanoparticles (CS BNPs).
# Feed different parameters into the LAMMPS input template file genBNPCS.in
# Author: Jonathan Yik Chang Ting
# Date: 29/10/2020

SUBSET_DIR="/scratch/q27/jt5911/iterative-archetypal-analysis/data/subsetsPickles"
OUTPUT_DIR="/scratch/q27/jt5911/iterative-archetypal-analysis/data/outputsPickles"
SCRIPT_PATH="/scratch/q27/jt5911/iterative-archetypal-analysis/PIAA.py"
SUBSET_FILES=($(ls $SUBSET_DIR/*.pickle))

python3 PIAA.py splitData
process_files() {
    local start=$1
    local end=$2
    for ((i=start; i<end; i++)); do
        python3 $SCRIPT_PATH ${SUBSET_FILES[$i]}
    done
}
python3 PIAA.py combineResults
echo -e "ALL DONE!"
