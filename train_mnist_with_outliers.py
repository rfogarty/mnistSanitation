
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torchvision as tv
import torchvision.transforms as tr
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
import numpy as np
import argparse
import pandas as pd
import os
#from showExamples import showMosaic
from MNISTSubset import *
MNIST_UNSAFE_DL()

def processCommandLine() :

    testn=0
    trainOutlierRatio=0.0
    testOutlierRatio=0.0
    trainFlipRatio=0.0
    testFlipRatio=0.0

    parser = argparse.ArgumentParser(description='Train Neural Net for some split.')

    parser.add_argument('-e','--epochs',dest='epochs',type=int,required=False,default=15,
                        help='Maximum number of epochs to train',metavar='EPOCHS')
    parser.add_argument('-b','--batch',dest='batch',type=int,required=False,default=64,
                        help='Batch size for training and test',metavar='BATCH')
    parser.add_argument('--test',dest='test',type=int,required=False,default=testn,
                        help='Test number for repeated tests (and unique archival)',metavar='TESTN')
    parser.add_argument('--trainOutliers',dest='trainOutliers',type=float,required=False,default=trainOutlierRatio,
                        help='Ratio of training outliers (or OOD)',metavar='TRAINOUT')
    parser.add_argument('--trainFlips',dest='trainFlips',type=float,required=False,default=trainFlipRatio,
                        help='Ratio of training label flips',metavar='TRAINFLIP')
    parser.add_argument('--testOutliers',dest='testOutliers',type=float,required=False,default=testOutlierRatio,
                        help='Ratio of testing outliers (or OOD)',metavar='TESTOUT')
    parser.add_argument('--testFlips',dest='testFlips',type=float,required=False,default=testFlipRatio,
                        help='Ratio of testing label flips',metavar='TESTFLIP')
    parser.add_argument('--trainData',dest='trainData',type=str,required=False,default=None,
                        help='Replica of prior randomized training data',metavar='TRAINFILE')
    parser.add_argument('--testData',dest='testData',type=str,required=False,default=None,
                        help='Replica of prior randomized testing data',metavar='TESTFILE')
    parser.add_argument('-m','--model',dest='model',type=str,required=False,default=None,
                        help='Saved model weights (if given, network will not retrain)',metavar='MODEL')
    parser.add_argument('--trainResults',dest='trainResults',type=str,required=False,default=None,
                        help='CSV File in which to archive training data results (typically ening in .csv)',metavar='TRAINRES')
    parser.add_argument('--testResults',dest='testResults',type=str,required=False,default=None,
                        help='CSV File in which to archive testing data results (typically ening in .csv)',metavar='TESTRES')
    parser.add_argument('--flipTraining',dest='flipTraining',type=str,required=False,default=None,
                        help='Flip training labels',metavar='FLIPTRAINING')
    parser.add_argument('--flipTesting',dest='flipTesting',type=str,required=False,default=None,
                        help='Flip testing labels',metavar='FLIPTESTING')
    parser.add_argument('--blockTraining',dest='blockTraining',type=str,required=False,default=None,
                        help='Block training samples',metavar='BLOCKTRAINING')
    parser.add_argument('--blockTesting',dest='blockTesting',type=str,required=False,default=None,
                        help='Block testing samples',metavar='BLOCKTESTING')
    parser.add_argument('--trainingLimitRatio',dest='trainingLimit',type=float,required=False,default=1.0,
                        help='Ratio of data to train on',metavar='TRAINLIMIT')
    parser.add_argument('--justGenData',dest='justGenData',action=argparse.BooleanOptionalAction,default=False,
                        help='Whether to exit early after generating randomized datasets')
    parser.add_argument('--skipTest',dest='skipTest',action=argparse.BooleanOptionalAction,default=False,
                        help='Skip testing and printing metrics')

    args=parser.parse_args()
    print(f'Configuration Arguments:\n{args}\n')
    
    return args


def loadDatasets(args) :

    #breakpoint()
    transform = tr.Compose([tr.ToTensor(),tr.Normalize((0.5), (0.5))])
    if args.trainData is not None :
        train_dataset = MNISTSubset(root='./data', train=True, transform=transform, download=True, modFile=args.trainData,flipFile=args.flipTraining,subset=args.blockTraining)
    else :
        train_dataset = MNISTSubset(root='./data', train=True, transform=transform, download=True, outlierRatio=args.trainOutliers,artificialError=args.trainFlips)
        filename = f'TrainingFlipsTest{args.test}_{args.trainOutliers}_{args.testOutliers}_{args.trainFlips}_{args.testFlips}.pkl'
        train_dataset.save(filename)

    if args.testData is not None :
        test_dataset = MNISTSubset(root='./data', train=False, transform=transform, download=True, modFile=args.testData,flipFile=args.flipTesting,subset=args.blockTesting)
    else :
        test_dataset = MNISTSubset(root='./data', train=False, transform=transform, download=True, outlierRatio=args.testOutliers,artificialError=args.testFlips)
        filename = f'TestingFlipsTest{args.test}_{args.trainOutliers}_{args.testOutliers}_{args.trainFlips}_{args.testFlips}.pkl'
        test_dataset.save(filename)

    train_loader = t.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    test_loader = t.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=True)

    return (train_loader,test_loader)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        #breakpoint()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = t.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = t.flatten(x)
        return x


def withPrecision(number, precision):
    """Prints a floating-point number with the specified precision."""
    format_string = f"{{:.{precision}f}}"
    return format_string.format(number)


def percent(number,digits=2) :
    p = np.round(100*(10**digits)*number)/(10**digits)
    return f'{withPrecision(p,digits)}'


def reorder(arr,indcs) :
    arr = np.array(arr)
    arr2 = arr.copy()
    arr[indcs] = arr2
    return arr


def testMNIST(net,device,dataloader,dataname='test',archive=None) :
    correct = 0
    total = 0
    y_true = list()
    y_pred = list()
    y_score = list()
    y_indcs = list()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with t.no_grad():
        debugging=False
        #print('Copy and run: debugging=True')
        #breakpoint()
        for data in dataloader:
            inputs, labels, indices = data[0].to(device), data[1].to(device), data[2]
            #if debugging :
            #    print(f'labels: {labels}')
            #    showMosaic(inputs)
            #    debugging=False
            # calculate outputs by running images through the network
            predictions = net(inputs)
            # the class with the highest energy is what we choose as prediction
            #_, predicted = torch.max(outputs.data, 1)
            outputs = t.round(predictions)
            #total += labels.size(0)
            #correct += (outputs == labels).sum().item()
            y_score.extend(predictions.cpu())
            y_pred.extend(outputs.cpu())
            y_true.extend(labels.cpu())
            y_indcs.extend(indices)
    
    #breakpoint()
    # Put the data back in its original non-shuffled order
    y_score = reorder(y_score,y_indcs)
    y_pred = reorder(y_pred,y_indcs)
    y_true = reorder(y_true,y_indcs)
    # And lastly for verification, reorder the indices!
    y_indcs = reorder(y_indcs,y_indcs)
    
    accuracy = percent(accuracy_score(y_true,y_pred))
    f1 = percent(f1_score(y_true,y_pred))
    auc = roc_auc_score(y_true,y_score)
    #print(f'Accuracy of the network on the {total} {dataname} images: {accuracy1}%')
    print(f'{dataname} metrics: {accuracy}%,{f1}%,{auc:.4f}')

    if archive is not None :
        if os.path.exists(archive) :
            df = pd.read_csv(archive,index_col=False)
        else :
            df = pd.DataFrame({'Index':y_indcs,'Truth':y_true})
        num = int((df.columns.size - 2)/2)
        df.insert(df.columns.size,f'Prob{num}',y_score)
        df.insert(df.columns.size,f'Pred{num}',y_pred)
        df.to_csv(archive,index=False)


def trainMNIST(device,dataloader,args) :

    net = Net()
    net.to(device)
    
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = opt.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # First determine the maximum number of batches that will be used to train upon
    # Limiting the amount of data to train each model can add diversity to ensembles.
    batches = 0
    for d in dataloader:
        batches = batches + 1
    maxBatches = int(np.floor(batches * args.trainingLimit))
    print(f'INFO: Only training first {maxBatches} of {batches} total batches.')
    
    for epoch in range(args.epochs):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            if i < maxBatches :
                # get the inputs; data is a list of [inputs, labels]
                #inputs, labels = data
                inputs, labels = data[0].to(device), data[1].to(device)
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward + backward + optimize
                outputs = net(inputs)
                #breakpoint()
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
    
                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
    
    print('Finished Training')
    return net

if __name__ == "__main__" :

    args = processCommandLine()

    # First set up device to run network
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    #device = t.device('cpu')
    print(f'device:{device}')

    trainloader,testloader = loadDatasets(args)

    if args.justGenData :
        print('Exiting early...')
        exit(0)

    # Train network if model is not passed
    if args.model is None :
        net = trainMNIST(device,trainloader,args)
        filename = f'modelTest{args.test}_{args.trainOutliers}_{args.testOutliers}_{args.trainFlips}_{args.testFlips}.ptm'
        t.save(net.state_dict(), filename)
    else :
        net = Net()
        net.load_state_dict(t.load(args.model, weights_only=True))
        net.to(device)

    # And Test
    if not args.skipTest :
        testMNIST(net,device,trainloader,dataname='training',archive=args.trainResults)
        testMNIST(net,device,testloader,dataname='testing',archive=args.testResults)


