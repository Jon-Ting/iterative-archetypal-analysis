from os import chdir, getcwd
import pickle

import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

chdir('/scratch/q27/jt5911/iterative-archetypal-analysis/iaa/docs')
import iaa
from iaa.piaa import ArchetypalAnalysis, subsetSplit, submitAAjobs, runAA, fitPIAA

FIG_SIZE, DPI = (7, 4), None
CONT_PALETTE, QUAL_PALETTE = 'viridis', 'tab10'
sns.set_style('ticks')

# General variables
DATA_DIR = f"{getcwd()}/data"
RANDOM_STATE = 42
NUM_DIM_MANIFOLD = 2
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

# Directory paths
SUBSETS_PICKLES_PATH = 'data/subsetsDataPKLs'
OUTPUTS_PICKLES_PATH = 'data/subsetsOutputsPKLs'
FIGS_DIR_PATH = 'figs'
JOBSCRIPTS_DIR_PATH = "jobScripts"
AA_SCRIPT_PATH = '/scratch/q27/jt5911/iterative-archetypal-analysis/iaa/src/iaa/runAA.py'

# Set random seed
np.random.seed(RANDOM_STATE)


def runPIAA2(Xscaled, numArchetypes=4, numSubset=4, onlyZ=False,
             robust=False, shuffle=True, randomState=RANDOM_STATE, 
             datasetType='uniformSquare', verbose=False, splitRunTime=0.0):
    AA = fitPIAA(Xscaled, nArchetypes=numArchetypes, numSubset=numSubset, 
                 dataName=datasetType, outputsPicklesPath=OUTPUTS_PICKLES_PATH, 
                 shuffle=shuffle, robust=robust, onlyZ=onlyZ, C=0.0001, tolerance=TOLERANCE, 
                 computeXtX=COMPUTE_XTX, stepsFISTA=STEPS_FISTA, stepsAS=STEPS_AS, randominit=RANDOM_INIT, 
                 randomState=randomState, numThreads=NUM_JOBS, splitRunTime=splitRunTime, verbose=verbose)
    if verbose:
        print(f"Parallel iterative archetypal analysis {numSubset} subsets execution time: {AA.runTime:.6f} s\n")
    return AA



# Load data
with open(f"{DATA_DIR}/BNPFeatsDF.pkl", 'rb') as f:
    BNPFeatsDF = pickle.load(f)
eleComb = 'BNP'

# Standardise data
sc = StandardScaler()
Xscaled = sc.fit_transform(BNPFeatsDF.iloc[:, :-1])

numSubset = 100
figNamePrefix = f"{eleComb}{numSubset}S_randomSplit"

# Debugging
# numArchetypesRange = range(100, 501, 100)
numArchetypesRange = range(5000, 5001, 100)
with open(f"{OUTPUTS_PICKLES_PATH}/{figNamePrefix}_splitRunTimes.pkl", 'rb') as f:
    splitRunTimes = pickle.load(f)
    
AAs = []
with open(f"{OUTPUTS_PICKLES_PATH}/{figNamePrefix}_5000As_PIAAs.pkl", 'rb') as f:
    oldAAs = pickle.load(f)
for (i, numArchetypes) in enumerate(numArchetypesRange):
    # Obtain the final results from the outcomes of the individual archetypal analysis on the subsets
    #AA = runPIAA2(Xscaled, numArchetypes=numArchetypes, numSubset=numSubset, onlyZ=False,
    #              robust=ROBUST, shuffle=SHUFFLE, randomState=RANDOM_STATE, 
    #              datasetType=f"{figNamePrefix}500As", verbose=VERBOSE, splitRunTime=splitRunTimes[i])
                  #datasetType=f"{figNamePrefix}{numArchetypes}As", verbose=VERBOSE, splitRunTime=splitRunTimes[i])
    AA = oldAAs[i]
    AA._extractCloseMatch()
    AAs.append(AA)

with open(f"{OUTPUTS_PICKLES_PATH}/{figNamePrefix}_5000As_PIAAs.pkl", 'wb') as f:
    pickle.dump(AAs, f)

print('DONE!')
