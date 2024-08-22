from os import chdir, getcwd
import pickle
import sys

import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

chdir('/scratch/q27/jt5911/iterative-archetypal-analysis/iaa/docs')
import iaa
from iaa.piaa import ArchetypalAnalysis, subsetSplit, submitAAjobs, fitPIAA

FIG_SIZE, DPI = (7, 4), None
CONT_PALETTE, QUAL_PALETTE = 'viridis', 'tab10'
sns.set_style('ticks')

# General variables
DATA_DIR = f"{getcwd()}/data"
RANDOM_STATE = 42
VERBOSE = True
NUM_JOBS = 48

# Variables specific for splitting data into subsets
SHUFFLE = True  # Whether to shuffle data samples before splitting.

# Variables specific for archetypes.py functionalities
C = 0.0001  # constraint coefficient to ensure that the summation of alpha and beta equals to 1.

# Variables specific for the archetypal analysis algorithm provided by the SPArse Modeling Software (SPAMS) package
ROBUST = False  # Whether to use modified AA algorithm to deal with outliers
TOLERANCE = 0.001  # Threshold to decide when to stop optimisation.
COMPUTE_XTX = False  # Whether to compute X^T * X
STEPS_FISTA = 3  # Number of steps for fast iterative shrinkage-thresholding algorithm
STEPS_AS = 50  # Number of steps for active set algorithm
RANDOM_INIT = False  # Whether to initialise the archetypes randomly

# Set random seed
np.random.seed(RANDOM_STATE)


SUBSETS_PICKLES_PATH = 'data/subsetsDataPKLs'
OUTPUTS_PICKLES_PATH = 'data/subsetsOutputsPKLs'
FIGS_DIR_PATH = 'figs'
JOBSCRIPTS_DIR_PATH = "jobScripts"
AA_SCRIPT_PATH = '/scratch/q27/jt5911/iterative-archetypal-analysis/iaa/src/iaa/runAA.py'


def genData(n, m):
    feats = [np.random.uniform(-10, 10, n) for i in range(m)]
    X = np.array([[feat[i] for i in range(n)] for feat in feats])
    sc = StandardScaler()
    Xscaled = sc.fit_transform(X.T)
    return Xscaled


def runRAA(Xscaled, numArchetypes=4, randomState=42):
    AAreference = ArchetypalAnalysis(nArchetypes=numArchetypes, iterative=False, robust=ROBUST, onlyZ=True,
                                     C=C, tolerance=TOLERANCE, computeXtX=COMPUTE_XTX, stepsFISTA=STEPS_FISTA, stepsAS=STEPS_AS, 
                                     randominit=RANDOM_INIT, randomState=randomState, numThreads=NUM_JOBS)
    AAreference.fit(Xscaled)
    return AAreference


def runPIAA1(Xscaled, numArchetypes=4, numSubset=4, shuffle=True, randomState=42, datasetType='uniformSquare', 
             project='q27', queue='normal', numCPUs=48, wallTime='01:00:00', mem=190, jobFS=100, 
             email='Jonathan.Ting@anu.edu.au', verbose=False):
    splitRunTime = subsetSplit(Xscaled, nSubsets=numSubset, dataName=datasetType, subsetsPicklesPath=SUBSETS_PICKLES_PATH,
                               shuffle=shuffle, randomState=randomState, verbose=verbose)
    submitAAjobs(nArchetypes=numArchetypes, dataName=datasetType, 
                 jobscriptsDirPath=JOBSCRIPTS_DIR_PATH, 
                 subsetsPicklesPath=SUBSETS_PICKLES_PATH, 
                 outputsPicklesPath=OUTPUTS_PICKLES_PATH, 
                 AAscriptPath=AA_SCRIPT_PATH,
                 project=project, queue=queue, numCPUs=numCPUs, wallTime=wallTime, mem=mem, jobFS=jobFS, email=email,
                 verbose=verbose)
    return splitRunTime


def runPIAA2(Xscaled, numArchetypes=4, numSubset=4, onlyZ=True,
             robust=False, shuffle=True, randomState=42, 
             datasetType='uniformSquare', verbose=False, splitRunTime=0.0):
    AAiterative = fitPIAA(Xscaled, nArchetypes=numArchetypes, numSubset=numSubset, 
                          dataName=datasetType, outputsPicklesPath=OUTPUTS_PICKLES_PATH, 
                          shuffle=shuffle, robust=robust, onlyZ=onlyZ, C=C, tolerance=TOLERANCE, 
                          computeXtX=COMPUTE_XTX, stepsFISTA=STEPS_FISTA, stepsAS=STEPS_AS, randominit=RANDOM_INIT, 
                          randomState=randomState, numThreads=NUM_JOBS, splitRunTime=splitRunTime, verbose=verbose)
    if verbose:
        print(f"Parallel iterative archetypal analysis {numSubset} subsets execution time: {AAiterative.runTime:.3f} s\n")
    return AAiterative


if __name__ == '__main__':
    # Global variable
    NUM_ITERATION = 30
    NUM_FEATURES = 1000
    SAMPLE_ARCH_RATIO = 10
    
    # User input
    analysis = sys.argv[1]
    task = sys.argv[2]
    numSamples = int(sys.argv[3])
    numCPUs = int(sys.argv[4])
    wallTime = sys.argv[5]

    # Derived variables
    numArchetypes = numSamples // SAMPLE_ARCH_RATIO
    numSubset = SAMPLE_ARCH_RATIO
    Xscaled = genData(n=numSamples, m=NUM_FEATURES)

    # Load or initialise splitRunTimes
    if task == 'PIAA1':
        splitRunTimes = []
    elif task == 'PIAA2':
        with open(f"{OUTPUTS_PICKLES_PATH}/{analysis}_{numSamples}_splitRunTimes.pkl", 'rb') as f:
            splitRunTimes = pickle.load(f)

    for i in range(NUM_ITERATION):
        if task == 'RAA':
            AA = runRAA(Xscaled, numArchetypes=numArchetypes, randomState=RANDOM_STATE + i)
            print(f"Run {i}: {AA.runTime:.3f} s")
        elif task == 'PIAA1':
            splitRunTime = runPIAA1(Xscaled, numArchetypes=numArchetypes, numSubset=numSubset, 
                                    shuffle=SHUFFLE, randomState=RANDOM_STATE, datasetType=f"{analysis}_{numSamples}_Run{i}", 
                                    numCPUs=numCPUs, wallTime=wallTime, mem=190, jobFS=1, verbose=VERBOSE)
            splitRunTimes.append(splitRunTime)
            print(f"Run {i} split time: {splitRunTime:.3f} s")
        elif task == 'PIAA2':
            AA = runPIAA2(Xscaled, numArchetypes=numArchetypes, numSubset=numSubset, onlyZ=True,
                          robust=ROBUST, shuffle=SHUFFLE, randomState=RANDOM_STATE, 
                          datasetType=f"{analysis}_{numSamples}_Run{i}", verbose=VERBOSE, splitRunTime=splitRunTimes[i])
            print(f"Run {i}: {AA.runTime:.3f} s")
    if task == 'PIAA1':
        with open(f"{OUTPUTS_PICKLES_PATH}/{analysis}_{numSamples}_splitRunTimes.pkl", 'wb') as f:
            pickle.dump(splitRunTimes, f)

    print('DONE!')
