
from torchvision.datasets import MNIST
import ssl
import pandas as pd
import numpy as np
import random
import os
import pickle

# Unsafe operation to disable Cert Authority checks, to download MNIST dynamically (or as needed)
#  Note: hack required due to CA issues in WSL
def MNIST_UNSAFE_DL() :
    ssl._create_default_https_context = ssl._create_unverified_context

class MNISTMods :

    def __init__(self,indices,bad_indices,flipped_indices,outlierRatio,artificialError) :
        self.indices = indices
        self.bad_indices = bad_indices
        self.flipped_indices = flipped_indices
        self.outlierRatio = outlierRatio
        self.artificialError = artificialError


class MNISTSubset(MNIST):

    def __artificialLabelError__(self) :
        if self.indices2Flip is not None and not self.suppressArtificialErrors :
            print(f'Artificially flipping {len(self.indices2Flip)} labels')
            for i in self.indices2Flip :
                # This is an extremely fast and clever way to flip binary labels 1 and 7: 1->-7->7, 7->-1->1
                self.targets[i] = int(-1*(self.targets[i] - 8))


    def __labelCorrection__(self,flipFile) :
        if flipFile is not None and os.path.exists(flipFile) :
            df = pd.read_csv(flipFile,index_col=False)
            print(f'Attempting to correct {len(df)} labels')
            for i in df['Index'] :
                self.targets[i] = int(-1*(self.targets[i] - 8))

    def __subsetOfSamples__(self,subset,invertBlock) :
        if subset is not None and os.path.exists(subset) :
            df = pd.read_csv(subset,index_col=False)
            toBlock = np.array([i for i in df['Index']])
            if invertBlock :
                print(f'Keeping {len(df)} of {len(self.indices)} samples')
                goodIndices = np.array([i for i in np.arange(len(self.indices)) if i in toBlock])
            else :
                print(f'Removing {len(df)} of {len(self.indices)} samples')
                goodIndices = np.array([i for i in np.arange(len(self.indices)) if i not in toBlock])
            self.data = self.data[goodIndices]
            self.targets = self.targets[goodIndices]


    def __init__(self,root,train=True,transform=None,targetTransform=None,download=False,modFile=None,outlierRatio=0,artificialError=0,
                 flipFile=None,subset=None,invertBlock=False,labelizeImage=False,suppressArtificialErrors=False):

        super().__init__(root, train=train, transform=transform, target_transform=targetTransform, download=download)
        self.indices2Flip = None
        self.labelizeImage = labelizeImage
        self.suppressArtificialErrors = suppressArtificialErrors
        if modFile is not None :
            # Fail fast, if modFile does not exist, allow the following to fail.
            with open(modFile,'rb') as f :
                mods = pickle.load(f)
                self.indices = mods.indices
                self.bad_indices = mods.bad_indices
                self.indices2Flip = mods.flipped_indices
                self.outlierRatio = mods.outlierRatio
                self.artificialError = mods.artificialError
                self.targets[self.bad_indices[0:int(len(self.bad_indices)/2)]] = 1
                self.targets[self.bad_indices[int(len(self.bad_indices)/2):]] = 7
                
        else :
            self.outlierRatio = outlierRatio
            self.artificialError = artificialError
            print(f'Total MNIST data: {len(self.targets)}')

            # Start by getting all indices with 1,7
            classes_to_keep = [1, 7]
            self.indices = [i for i, label in enumerate(self.targets) if label in classes_to_keep]

            if outlierRatio > 0 :
                #breakpoint()
                print(f'Adding {outlierRatio*100}% OOD into data set')
                classes_to_discard = [0,2,3,4,5,6,8,9]
                self.bad_indices = [i for i, label in enumerate(self.targets) if label in classes_to_discard]
                numOutliers = int(np.round(len(self.indices)*outlierRatio/(1-outlierRatio)))
                if numOutliers > len(self.bad_indices) :
                    numOutliers = len(self.bad_indices)
                    # if we do not want to lower number of training samples, then we can just print the following warning, and continue
                    #print(f'Unable to support outlierRatio={outlierRatio}, data only supports {len(self.bad_indices)/(len(self.bad_indices)+len(self.indices))}')
                    # Comment out the following lines if we do NOT want to reduce good data samples...
                    numGood=int(np.round(numOutliers*(1-outlierRatio)/outlierRatio))
                    random.shuffle(self.indices)
                    oldSize = len(self.indices)
                    self.indices = self.indices[0:numGood]
                    print(f'Unable to support outlierRatio={outlierRatio} w/o shrinking IID samples data from {oldSize} to {len(self.indices)}')
                np.random.shuffle(self.bad_indices)
                self.bad_indices = self.bad_indices[0:numOutliers]
                # Assign half of these to 1 and the other half to 7
                self.targets[self.bad_indices[0:int(len(self.bad_indices)/2)]] = 1
                self.targets[self.bad_indices[int(len(self.bad_indices)/2):]] = 7
                self.indices.extend(self.bad_indices)
        
        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]
        print(f'Pruned MNIST data: {len(self.targets)}')
        if modFile is None and artificialError > 0 :
            # First create random set of indices
            indices = np.arange(len(self.targets))
            np.random.shuffle(indices)
            num2Flip = int(len(self.targets) * artificialError)
            print(f'Flipping {artificialError*100}% or {num2Flip} of {len(indices)} labels')
            self.indices2Flip=indices[0:num2Flip]

        # Flip randomly selected labels if so directed.
        self.__artificialLabelError__()

        # Flip questionable labels if so directed.
        self.__labelCorrection__(flipFile)

        # If there are any samples to remove (or "block"), we can do that here
        self.__subsetOfSamples__(subset,invertBlock)

        # Finally let's normalize the targets (labels): 1->0->0, 7->6->1
        self.targets = (self.targets - 1)/6


    def save(self,filename) :
        with open(filename,'wb') as f :
            mods = MNISTMods(self.indices,self.bad_indices,self.indices2Flip,self.outlierRatio,self.artificialError)
            pickle.dump(mods,f)


    def __getitem__(self, index):
        image, label = super(MNISTSubset,self).__getitem__(index)
        if self.labelizeImage :
            if label == 0 :
                image = image * -1
        return image, label, index


    def __len__(self):
        return len(self.targets)


