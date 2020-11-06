# 使用"训练好的LeNet模型(使用CrossEntropyLoss、dirichlet损失函数，基于前五个类训练的)"
# 进行proErr, uncertainty
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import matplotlib.pyplot as plt 
from utils.model import CifarNN
from utils.countNp import GetUncFre
from utils.assit import *

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

print ("Loading Cifar10 Test Dataset...")
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testsetCifar = torchvision.datasets.CIFAR10(root='../Dataset/cifar', train=False, download= True, transform=transform)
testLoaderCifar = torch.utils.data.DataLoader(testsetCifar, batch_size=batchSize, shuffle=True, num_workers=0)

lamda = 0.1
classNumber = 10
modelName = 'EDL' #'CNN', 'EDL', CNNEDL
datasetNames = ['CIFAR10', 'SVHN']
lineStyle = {'CIFAR10': '-.k', 'SVHN': '-r'}
testLoaders = {'CIFAR10': testLoaderCifar, 'SVHN': testLoaderSvhn}
device = GetDevice()

model = CifarNN(classNumber)
ToDevice(model, device)

if modelName == 'CNNEDL':
    model.load_state_dict(torch.load('../DictState/Cifar10-CNN.pt'))
else:
    model.load_state_dict(torch.load('../DictState/Cifar10-{}.pt'.format(modelName)))

for datasetName in datasetNames:
    print("{} Testing, model trained by {}, postfixed DD...".format(datasetName, modelName))
    testLoader = DeviceDataLoader(testLoaders[datasetName], device)

    model.eval()
    accNum = 0.0
    uList = []
    
    for j, (images, labels) in enumerate(testLoader):
        outputs = model(images) 
        if modelName == 'CNN':
            evidence = F.softmax(outputs, dim= 1)
            proErr = 1 - torch.max(evidence, dim= 1).values
            uList.append(proErr.detach())
            accNum += torch.sum(torch.argmax(evidence, dim= 1) == labels).item()
        elif modelName == 'CNNEDL':
            evidence = F.relu(outputs)
            alpha = torch.exp(lamda * evidence)
            #alpha = evidence + 1
            alphaAcc = torch.sum (alpha, dim = 1, keepdims = True)
            prioriP = alpha / alphaAcc
            u = torch.squeeze(classNumber / alphaAcc) 
            uList.append(u.detach())
            accNum += torch.sum(torch.argmax(prioriP, dim= 1) == labels).item()
            torch.save(prioriP, '../Outputs/{}-CNNEDL.out'.format(datasetName))

        else:
            evidence = F.relu(outputs)
            alpha = torch.exp(lamda * evidence)
            #alpha = evidence + 1
            alphaAcc = torch.sum (alpha, dim = 1, keepdims = True)
            prioriP = alpha / alphaAcc
            u = torch.squeeze(classNumber / alphaAcc) 
            uList.append(u.detach())
            accNum += torch.sum(torch.argmax(prioriP, dim= 1) == labels).item()
            torch.save(prioriP, '../Outputs/{}-EDL.out'.format(datasetName))

    acc =  accNum / len(testLoader.dl.dataset)
    print("{} Test acc= {}".format(datasetName, acc)) 

    uncer = torch.cat(uList, 0)
    uncKeyFre = GetUncFre(uncer)
    plt.title(modelName)
    plt.xlabel('Uncer')
    plt.ylabel('Freq')
    plt.plot(list(i / uncKeyFre['decNum'] for i  in uncKeyFre['key']) , list(i / len(testLoader.dl.dataset) for i in uncKeyFre['fre']), lineStyle[datasetName], label= datasetName)
    plt.legend()
plt.show()

