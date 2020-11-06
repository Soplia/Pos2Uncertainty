# 使用"训练好的LeNet模型(使用CrossEntropyLoss、dirichlet损失函数，基于前五个类训练的)"
# 进行proErr, uncertainty
import torch
import numpy as np 
import matplotlib.pyplot as plt 
from utils.model import WideResNet
from utils.CalCriteria import show_performance, get_measures
from utils.CalScores import GetScores, get_ood_scores_odin, get_Mahalanobis_score, sample_estimator
from plot.ROcPlot import PlotROC
from utils.assit import *
import sklearn.metrics as sk
from torch.autograd import Variable
batchSize = 100
print ("Loading SVHN Test Dataset...")
# torch.Size([26032, 3, 32, 32])
svhnFeature = torch.load('../Dataset/svhn/SVHNpicFeature').type(torch.FloatTensor)
#  torch.Size([26032, 10])
svhnOnehotLabel = torch.tensor(torch.load('../Dataset/svhn/SVHNpicLabel')).type(torch.FloatTensor)
# torch.Size([26032])
svhnLabel = torch.squeeze(torch.argmax(svhnOnehotLabel, dim = 1))
testsetSvhn = torch.utils.data.TensorDataset(svhnFeature, svhnLabel)
testLoaderSvhn = torch.utils.data.DataLoader(testsetSvhn, batch_size=batchSize, shuffle=True, num_workers=0)


oodDatasetNames = ['SVHN'] # 'CIFAR100', 'LSUN', 'TEXTURE', 'PLACE365', 'CIFAR100', 'LSUN', 
                   #'TEXTURE', 'PLACE365'
scoreTypes = ['Mahalanobis', 'ODIN'] #'Our', 'Energy','Softmax', 'Dirichlet'

modelName = 'WideResNet' 
droprate = 0.3
classNumber = 10
totalNumLayers = 40
widenFactor = 2
model = WideResNet(totalNumLayers, classNumber, widenFactor, dropRate= droprate)
model.load_state_dict(torch.load('../DictState/pretrained/cifar10_wrn_pretrained_epoch_99.pt'))
model.eval()
device = GetDevice()
ToDevice(model, device)
cifar10Outputs = torch.load('../Outputs/{}-Cifar10-Cifar10CNN-Outputs'.format(modelName))
inScore = GetScores(cifar10Outputs, method= '{}'.format('Our'))

for oodName in oodDatasetNames:
    for scoreType in scoreTypes:
        print("{} score is calculating...".format(scoreType))
        if scoreType == 'Mahalanobis':
            temp_x = torch.rand(2,3,32,32)
            temp_x = Variable(temp_x)
            temp_x = temp_x.cuda()
            temp_list = model.feature_list(temp_x)[1]
            num_output = len(temp_list)
            feature_list = np.empty(num_output)
            count = 0
            for out in temp_list:
                feature_list[count] = out.size(1)
                count += 1

            print('get sample mean and covariance', count)
            sample_mean, precision = sample_estimator(model, classNumber, feature_list, testLoaderSvhn) 

            oodScore = get_Mahalanobis_score(model, testLoaderSvhn, classNumber, sample_mean, precision, 
                                  layer_index= 1, magnitude= 0, num_batches= classNumber // batchSize, in_dist=False)
            oodScore = torch.from_numpy(oodScore)
        elif scoreType == 'ODIN':
            oodScore = get_ood_scores_odin(testLoaderSvhn, model, batchSize, 26032, T= 10, noise= 0, in_dist=False)
            oodScore = torch.from_numpy(oodScore)
        else:
            oodScore = GetScores(oodOutputs, method= scoreType)
        #show_performance(inScore, oodScore, method_name=scoreType)
        auroc, aupr, fpr95 = get_measures(inScore, oodScore)
        print ('&{}& {}& {}& {} \\\\'.format(scoreType, str(auroc)[0: 4], str(aupr)[0: 4], str(fpr95)[0: 4])) 

        if oodName == 'SVHN':
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
            plt.title('ROC for CIFAR10 (pos) & SVHN (neg)')
            plt.xlabel('FPR; {} score (pretrained)'.format(scoreType))
            plt.ylabel('TPR')
            plt.plot(fpr, tpr, linestyle= ':',  markerfacecolor= 'red', color = 'k', 
                     label='fpr95:{}, auroc:{}, aupr:{}'.format(str(fpr95)[0: 4], str(auroc)[:4], str(aupr)[:4])) 
            plt.fill_between(fpr, tpr, step="pre", color= 'darkgray', alpha=0.4)
            plt.axis([0, 1, 0, 1])
            plt.grid(alpha= 0.8)
            plt.legend()
            plt.savefig('../Figures/{}-CIFAR10-SVHN-{}'.format(modelName, scoreType))
            plt.show()
            

    print(r'\hline')
        


