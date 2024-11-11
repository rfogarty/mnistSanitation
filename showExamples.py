
import torch as t
import torchvision as tv
import torchvision.transforms as tr
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as pl
from MNISTSubset import *
MNIST_UNSAFE_DL()


def showMosaic(images,nrow=8,filename=None) :
    images = tv.utils.make_grid(images,nrow=nrow)
    images = images / 2 + 0.5;
    images = images.cpu().numpy()
    images = np.transpose(images,(1,2,0))
    pl.imshow(images)
    if filename is not None :
        pl.savefig(filename)
    else :
        pl.show()


def processMosaicCommandLine() :

    testn=0
    trainOutlierRatio=0.0
    testOutlierRatio=0.0
    trainFlipRatio=0.0
    testFlipRatio=0.0

    parser = argparse.ArgumentParser(description='Arguments to show example mosaic')

    parser.add_argument('-b','--batch',dest='batch',type=int,required=False,default=64,
                        help='Batch size for training and test',metavar='BATCH')
    parser.add_argument('--test',dest='test',type=int,required=False,default=testn,
                        help='Test number for repeated tests (and unique archival)',metavar='TESTN')
    parser.add_argument('--trainData',dest='trainData',type=str,required=False,default=None,
                        help='Replica of prior randomized training data',metavar='TRAINFILE')
    parser.add_argument('--testData',dest='testData',type=str,required=False,default=None,
                        help='Replica of prior randomized testing data',metavar='TESTFILE')
    parser.add_argument('--flipTraining',dest='flipTraining',type=str,required=False,default=None,
                        help='Flip training labels',metavar='FLIPTRAINING')
    parser.add_argument('--flipTesting',dest='flipTesting',type=str,required=False,default=None,
                        help='Flip testing labels',metavar='FLIPTESTING')
    parser.add_argument('--blockTraining',dest='blockTraining',type=str,required=False,default=None,
                        help='Block training samples',metavar='BLOCKTRAINING')
    parser.add_argument('--blockTesting',dest='blockTesting',type=str,required=False,default=None,
                        help='Block testing samples',metavar='BLOCKTESTING')
    parser.add_argument('--invertBlocked',dest='invertBlocked',action=argparse.BooleanOptionalAction,default=False,
                        help='Invert set of indices that are blocked from display')
    parser.add_argument('--shuffle',dest='shuffle',action=argparse.BooleanOptionalAction,default=False,
                        help='Shuffle data set in mosaics (default=False)')
    parser.add_argument('--labelEffect',dest='labelEffect',action=argparse.BooleanOptionalAction,default=False,
                        help='Recolor image according to label')
    parser.add_argument('--noArtificialErrors',dest='noArtificialErrors',action=argparse.BooleanOptionalAction,default=False,
                        help='Suppress artificial errors added to dataset')
    parser.add_argument('-s','--savePrefix',dest='savePrefix',type=str,required=False,default=None,
                        help='Instead of displaying interactively, save to file starting with prefix',metavar='SAVEPRE')

    args=parser.parse_args()
    print(f'Configuration Arguments:\n{args}\n')
    
    return args



def loadMNISTMods(args) :

    #breakpoint()
    transform = tr.Compose([tr.ToTensor(),tr.Normalize((0.5), (0.5))])
    train_loader = None
    test_loader = None
    if args.trainData is not None :
        train_dataset = MNISTSubset(root='./data', train=True, transform=transform, download=True,
                                    modFile=args.trainData,flipFile=args.flipTraining,
                                    subset=args.blockTraining,invertBlock=args.invertBlocked,labelizeImage=args.labelEffect,suppressArtificialErrors=args.noArtificialErrors)
        train_loader = t.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=args.shuffle)

    if args.testData is not None :
        test_dataset = MNISTSubset(root='./data', train=False, transform=transform, download=True,
                                   modFile=args.testData,flipFile=args.flipTesting,
                                   subset=args.blockTesting,invertBlock=args.invertBlocked,labelizeImage=args.labelEffect,suppressArtificialErrors=args.noArtificialErrors)
        test_loader = t.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=args.shuffle)

    return (train_loader,test_loader)


def buildMosaics(dataloader,dataname='test',nrow=8,prefixToSave=None) :
    # For this tool, always run it in debug mode
    for i,data in enumerate(dataloader):
        inputs, labels, indices = data[0], data[1], data[2]
        print(f'labels: {labels}')
        if prefixeToSave is not None :
            filename = f'{prefixToSave}_{dataname}_{i}.png'
            showMosaic(inputs,nrow=nrow,save=filename)
        else :
            showMosaic(inputs,nrow=nrow)

if __name__ == "__main__" :

    args = processMosaicCommandLine()

    trainloader,testloader = loadMNISTMods(args)

    if trainloader is not None :
        buildMosaics(trainloader,dataname='Training',nrow=int(np.sqrt(args.batch)),args.savePrefix)
    
    if testloader is not None :
        buildMosaics(testloader,dataname='Testing',nrow=int(np.sqrt(args.batch)),savePrefix)


