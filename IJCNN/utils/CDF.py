import torch
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def RatioEntropyMaxEntropy(entropys):
    maxEntropy = torch.max(entropys)
    avgRatio = torch.mean(entropys / maxEntropy)
    return avgRatio

def ShanEntropy(values):
    """
    value size= [batchSize, number of classes]
    """
    ## This is used to calculated probability
    #valueNormalized = value / float(np.sum(value))
    H = torch.empty(values.shape[0])
    for i, value in enumerate(values):
        value = value[value != 0]
        H[i] = -torch.sum(value* torch.log2(value))  
    return H

def VisualiseCDF(values, title= '', visDot= True):
    values = values.detach().numpy()
    ecdf = sm.distributions.ECDF(values)
    x = np.linspace(min(values), max(values))
    ##Calculate the corresponding cumulative density probability based on the data on the x-axis
    y = ecdf(x)

    plt.title(title)
    plt.xlabel('Entropy')
    plt.ylabel('Probability')
    ##Draw a ladder diagram
    #plt.step(x, y)
    plt.plot(x, y, color='k', linewidth=1.5, alpha=1.0)
    if visDot:
        yMain = ecdf(values)
        plt.scatter(values, yMain, c='r', s=20, alpha=0.5)
       
    plt.show()

def MiniDemo():
    values = torch.tensor([[1, 0, 0], [0.1, 0.4, 0.5], [.3, .6, 0.1]])
    H = ShanEntropy(values)
    ratio = RatioEntropyMaxEntropy(H)
    print (ratio)

datasetNames = ['CIFAR10', 'SVHN']
for datasetName in datasetNames:
    prob = torch.load('../Outputs/{}-EDL.out'.format(datasetName))
    entropys = ShanEntropy(prob)
    #VisualiseCDF(shanProb)
    ratio = RatioEntropyMaxEntropy(entropys)
    print ('Entropy ratio of {}= {}'.format(datasetName, ratio))

#models = ['CNNEDL', 'EDL']
#for model in models:
#    prob = torch.load('../Outputs/SVHN-{}.out'.format(model))
#    shanProb = ShanEntropy(prob)
#    VisualiseCDF(shanProb, title = model)
