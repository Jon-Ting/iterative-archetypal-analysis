#!/bin/bash
#PBS -P q27
#PBS -q normal
#PBS -l ncpus=48,walltime=20:00:00,mem=120GB,jobfs=5GB
#PBS -l storage=scratch/q27
#PBS -l wd
#PBS -M Jonathan.Ting@anu.edu.au
#PBS -m a

module load python3/3.10.4

python3 runIAA_BNP.py
