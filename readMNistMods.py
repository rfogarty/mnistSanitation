import pickle
import pandas as pd
import os

class MNISTMods :

    def __init__(self,indices,bad_indices,flipped_indices,outlierRatio,flipRatio) :
        self.indices = indices
        self.bad_indices = bad_indices
        self.flipped_indices = flipped_indices
        self.outlierRatio = outlierRatio
        self.flipRatio = flipRatio



def saveFlippedIID(mnistModFile,flippedIIDsFile) :
    mods = None
    with open(mnistModFile,'rb') as f :
        mods = pickle.load(f)
    
    numOODs = len(mods.bad_indices)
    numIIDs = len(mods.indices) - numOODs
    print(f'num IID samples: {numIIDs}')
    print(f'num OOD samples: {numOODs}')
    flippedIIDs = [i for i in mods.flipped_indices if i < numIIDs]
    
    df = pd.DataFrame({'Flipped':flippedIIDs})
    df.to_csv(flippedIIDsFile,index=False)


saveFlippedIID(os.environ['TRAINDATA'],'OriginalTrainingFlips.csv')
saveFlippedIID(os.environ['TESTDATA'],'OriginalTestingFlips.csv')

