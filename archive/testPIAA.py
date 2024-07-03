import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

from PIAA import *  # SUBSETS_PICKLES_PATH, OUTPUTS_PICKLES_PATH, RANDOM_STATE



if __name__ == '__main__':
    task = sys.argv[1]
    print(f"Task: {task}")

    NUM_SUBSETS = 4
    NUM_SAMPLES = 5000
    NUM_ARCHETYPES = 4
    
    # Generate data
    DATASET_TYPE = 'uniformSquare'
    x = np.random.uniform(-10, 10, NUM_SAMPLES)
    y = np.random.uniform(-10, 10, NUM_SAMPLES)
    X = np.array([[xi, yi] for xi, yi in zip(x, y)])
    sc = StandardScaler()
    Xscaled = sc.fit_transform(X)
    
    if task == 'subsetSplit':
        splitRunTime = subsetSplit(Xscaled, n_subsets=NUM_SUBSETS, dataName=DATASET_TYPE, subsetsPicklesPath=SUBSETS_PICKLES_PATH,
                                   shuffle=True, random_state=RANDOM_STATE, verbose=True)
    elif task == 'runAA':
        runAA(fName=sys.argv[2], n_archetypes=sys.argv[2], 
              outputsPicklesPath=OUTPUTS_PICKLES_PATH)
    elif task == 'fitPIAA':
        AA = fitPIAA(Xscaled, n_archetypes=NUM_ARCHETYPES, numSubset=NUM_SUBSETS, 
                     dataName=DATASET_TYPE, outputsPicklesPath=OUTPUTS_PICKLES_PATH, splitRunTime=splitRunTime)

    print('ALL DONE!')