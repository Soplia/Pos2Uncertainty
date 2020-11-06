# 使用"训练好的LeNet模型(使用CrossEntropyLoss、dirichlet损失函数，基于前五个类训练的)"
# 进行proErr, uncertainty
import torch
import numpy as np 
import matplotlib.pyplot as plt 
from utils.model import CifarNN
from utils.CalCriteria import show_performance, get_measures
from utils.CalScores import GetScores
from plot.ROcPlot import PlotROC
from utils.assit import *
import sklearn.metrics as sk

oodDatasetNames = ['SVHN' , 'CIFAR100', 'LSUN', 
                    'TEXTURE', 'PLACE365'
                   ] # 'CIFAR100', 'LSUN', 'TEXTURE', 'PLACE365', 'KMNIST', 'FMNIST'
scoreTypes = [ 'Our', 'Energy']#,'Softmax', 'Dirichlet'
Ts = [0.1, 0.5, 1, 5, 10, 50, 100]
isShow = False
modelName = 'WideResNet' 
cifar10Outputs = torch.load('../Outputs/{}-Cifar10-Cifar10CNN-Outputs'.format(modelName))
for T in Ts:
    print ("T= {}".format(T))
    for oodName in oodDatasetNames:
            
            oodOutputs = torch.load('../Outputs/{}-{}-Cifar10CNN-Outputs'.format(modelName, oodName))
            print('\multirow{2}{*}{', oodName, '}', end=' ')
            for scoreType in scoreTypes:
                #print("\t {} score is calculating...".format(scoreType))
                inScore = GetScores(cifar10Outputs, T, method= '{}'.format(scoreType))
                oodScore = GetScores(oodOutputs, T, method= scoreType)
                #show_performance(inScore, oodScore, method_name=scoreType)
                auroc, aupr, fpr95 = get_measures(inScore, oodScore)
                print ('&{}& {}& {}& {} \\\\'.format(scoreType, str(auroc)[0: 4], str(aupr)[0: 4], str(fpr95)[0: 4])) 

                if oodName == 'SVHN' and isShow == True:
                    bins= 200
                    inX = np.linspace(inScore.min(), inScore.max(), bins) 
                    inHist = torch.histc(inScore, bins=bins, min=inScore.min(), max= inScore.max())
                    oodX = np.linspace(oodScore.min(), oodScore.max(), bins) 
                    oodHist = torch.histc(oodScore, bins=bins, min=oodScore.min(), max= oodScore.max())
                    plt.figure(figsize=(10, 8))
                    plt.subplot(1, 2, 1)
                    plt.xlabel('{} score (pretrained); fpr95: {}'.format(scoreType, str(fpr95)[0: 4]))
                    plt.ylabel('Frequency') 
            
                    plt.plot(inX, inHist, color= 'r', label= 'in-domain (CIFAR10)')
                    plt.fill_between(inX, inHist, step="pre", color= 'darkgray', alpha=0.4)
            
                    plt.plot(oodX , oodHist, color= 'g', label= 'out-of-domain ({})'.format(oodName))
                    plt.fill_between(oodX, oodHist, step="pre", color= 'cornflowerblue', alpha=0.4)
            
                    plt.axis([min(inX.min(), oodX.min()), max(inX.max(), oodX.max()), min(inHist.min(), oodHist.min()), max(inHist.max(), oodHist.max())])
                    plt.grid(alpha= 0.8)
                    plt.legend()
                    #plt.savefig('../Figures/{}-CIFAR10-SVHN-{}'.format(modelName, scoreType))
                    #plt.show()

                    predictionProb = torch.cat((inScore, oodScore), 0)
                    realLabel = torch.cat((torch.zeros(inScore.shape[0]), torch.ones(oodScore.shape[0])), 0)
                    #0: in-domain; 1:out-of-domain
                    fpr, tpr, thresholds = sk.roc_curve(realLabel, predictionProb, pos_label= 0)
                    plt.subplot(1, 2, 2)
                    plt.title('T={} ROC for CIFAR10 (pos) & SVHN (neg)'.format(T))
                    plt.xlabel('FPR; {} score (pretrained)'.format(scoreType))
                    plt.ylabel('TPR')
                    plt.plot(fpr, tpr, linestyle= ':',  markerfacecolor= 'red', color = 'k', 
                             label='fpr95:{}, auroc:{}, aupr:{}'.format(str(fpr95)[0: 4], str(auroc)[:4], str(aupr)[:4])) 
                    plt.fill_between(fpr, tpr, step="pre", color= 'darkgray', alpha=0.4)
                    plt.axis([0, 1, 0, 1])
                    plt.grid(alpha= 0.8)
                    plt.legend()
                    plt.savefig('../Figures/T={}-{}-CIFAR10-SVHN-{}'.format(T,modelName, scoreType))
                    plt.show()
            print(r'\hline')
        


