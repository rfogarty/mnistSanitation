import pickle
import sys
from MNISTSubset import *

if len(sys.argv) != 2 :
    print(f'Run with: python {sys.argv[0]} <MNISTModsFile.pkl>')
    exit(1)

modFile=sys.argv[1]

with open(modFile,'rb') as f :
    mods = pickle.load(f)

mods2 = MNISTMods(mods.indices,mods.bad_indices,mods.flipped_indices,mods.outlierRatio,mods.flipRatio)

with open(modFile,'wb') as f :
    pickle.dump(mods2,f)

