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
print ("Loading Cifar10 Test Dataset...")
#testFeature = torch.from_numpy(np.transpose(np.load("../Dataset/cifar/cifar10-test-features.npy"), (0, 3, 1, 2)))
#testLabel = torch.squeeze(torch.from_numpy(np.load("../Dataset/cifar/cifar10-test-labels.npy"))).type(torch.LongTensor)
#testLabel = (testLabel + 5) % 10
#testset = torch.utils.data.TensorDataset(testFeature, testLabel)
#testLoader = torch.utils.data.DataLoader(testset, batch_size= batchSize, shuffle=False, num_workers=0)

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR10(root='../Dataset/cifar', train=False, download= True, transform=transform)
testLoader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=True, num_workers=0)

classNumber = 10
modelType = ['CNN', 'EDL', 'CNN-EDL'] #'
lineStyle = {'CNN': '-.k', 'EDL': '-r', 'CNN-EDL': '--b'}

model = CifarNN(classNumber)
device = GetDevice()
testLoader = DeviceDataLoader(testLoader, device)
ToDevice(model, device)

for  modelName in modelType:
    if modelName == 'CNN-EDL':
        model.load_state_dict(torch.load('../DictState/Cifar10-CNN.pt'))
    else:
        model.load_state_dict(torch.load('../DictState/Cifar10-{}.pt'.format(modelName)))
    print("{} Testing ...".format(modelName))
    model.eval()
    accNum = 0.0
    uList = []

    #predLabeList = []
    
    for j, (images, labels) in enumerate(testLoader):
        outputs = model(images) 

        if modelName == 'CNN':
            evidence = F.softmax(outputs, dim= 1)
            proErr = 1 - torch.max(evidence, dim= 1).values
            uList.append(proErr.detach())
            #predLabel = torch.squeeze(torch.argmax(evidence, dim= 1))
            #predLabeList.append(predLabel)
            accNum += torch.sum(torch.argmax(evidence, dim= 1) == labels).item()
        #elif modelName == 'CnnEdl':
        #    evidence = F.relu(outputs)
        #    alpha = evidence + 1
        #    alphaAcc = torch.sum (alpha, dim = 1, keepdims = True)
        #    prioriP = alpha / alphaAcc
        #    u = torch.squeeze(classNumber / alphaAcc) 
        #    uList.append(u)
        #    predLabel = torch.squeeze(torch.argmax(prioriP, dim= 1))
        #    predLabeList.append(predLabel)
        #    accNum += torch.sum(torch.argmax(prioriP, dim= 1) == labels).item()
        else:
            evidence = F.relu(outputs)
            alpha = evidence + 1
            alphaAcc = torch.sum (alpha, dim = 1, keepdims = True)
            prioriP = alpha / alphaAcc
            u = torch.squeeze(classNumber / alphaAcc) 
            uList.append(u.detach())
            #predLabel = torch.squeeze(torch.argmax(prioriP, dim= 1))
            #predLabeList.append(predLabel)
            accNum += torch.sum(torch.argmax(prioriP, dim= 1) == labels).item()

    #acc =  accNum / testLabel.shape[0]
    acc =  accNum / len(testLoader.dl.dataset)
    print("{} Test acc= {}".format(modelName, acc)) 

    uncer = torch.cat(uList, 0)
    uncKeyFre = GetUncFre(uncer)

    plt.xlabel('Uncer')
    plt.ylabel('Freq')
    plt.plot(list(i / uncKeyFre['decNum'] for i  in uncKeyFre['key']) , list(i / len(testLoader.dl.dataset) for i in uncKeyFre['fre']), lineStyle[modelName], label= modelName)
    plt.legend()
plt.show()

