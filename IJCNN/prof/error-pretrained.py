# 使用"训练好的LeNet模型(使用CrossEntropyLoss、dirichlet损失函数，基于前五个类训练的)"
# 进行proErr, uncertainty
import torch
import numpy as np 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
from utils.CalCriteria import show_performance, get_measures
from utils.CalScores import GetScores
from utils.assit import *
import sklearn.metrics as sk

#a = torch.tensor([5, 4, 3, 2, 1])
#b = torch.tensor([[5, 5], [4, 4], [3, 3], [2, 2], [1, 1]])
#print (b[torch.argsort(a)])

print ("Loading Cifar10 Test Dataset...")
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testsetCifar = torchvision.datasets.CIFAR10(root='../Dataset/cifar', train=False, download= True, transform=transform)

print ("Loading SVHN Test Dataset...")
svhnFeature = torch.load('../Dataset/svhn/SVHNpicFeature').type(torch.FloatTensor)
svhnOnehotLabel = torch.tensor(torch.load('../Dataset/svhn/SVHNpicLabel')).type(torch.FloatTensor)
svhnLabel = torch.squeeze(torch.argmax(svhnOnehotLabel, dim = 1))

print ("Loading LSUN dataset...")
lsunFeature = torch.load('../Dataset/lsun/LSUNpicFeature').type(torch.FloatTensor)
lsunOnehotLabel = torch.load('../Dataset/lsun/LSUNpicLabel').type(torch.FloatTensor)
lsunLabel = torch.squeeze(torch.argmax(lsunOnehotLabel, dim = 1))

print ("Loading Cifar100 Test Dataset...")
cifar100Feature = torch.load('../Dataset/cifar100/cifar100-test-features').type(torch.FloatTensor)
cifar100Label = torch.load('../Dataset/cifar100/cifar100-test-labels')

#print ("Loading TEXTURE Test Dataset...")
##Size([10000, 3, 32, 32])
#textureFeature = torch.load('../Dataset/texture/texture-test-features').type(torch.FloatTensor)[0:1300]
#textureTestset = torch.utils.data.TensorDataset(textureFeature)
#testLoaderTexture = torch.utils.data.DataLoader(textureTestset, batch_size=batchSize, shuffle=True, num_workers=0)

#print ("Loading Places365 Test Dataset...")
##  Size([10000, 3, 32, 32])
#places365Feature = torch.load('../Dataset/places365/place365-test-features').type(torch.FloatTensor)
#places365Testset = torch.utils.data.TensorDataset(places365Feature)
#testLoaderplaces365 = torch.utils.data.DataLoader(places365Testset, batch_size=batchSize, shuffle=True, num_workers=0)

oodDatasetNames = ['SVHN', 'LSUN', 'CIFAR100'] # 'KMNIST', 'FMNIST', 'PLACE365', 'TEXTURE', 
scoreTypes = ['Our', 'Energy','Softmax', 'Dirichlet']

inDomainDs = 'CIFAR10'
modelName = 'WideResNet' 
trainType= 'pretrained'
cifar10Outputs = torch.load('../Outputs/{}-{}-Cifar10-Outputs'.format(trainType, inDomainDs))
cifar10Labels = torch.tensor(testsetCifar.targets)
cifar10Omega = GetScores(cifar10Outputs, method= 'omega')
cifar10Mass = GetScores(cifar10Outputs, method= 'mass')
idx = torch.argsort(cifar10Omega)
cifar10Mass = cifar10Mass[idx]
cifar10Labels = cifar10Labels[idx]
cifar10Preds = torch.argmax(cifar10Mass, dim = 1)
rightClassfied = torch.cumsum(cifar10Preds == cifar10Labels, dim = 0)
totalSamples = torch.cumsum(torch.ones(rightClassfied.shape), dim = 0)
errorRate = (totalSamples - rightClassfied.type_as(totalSamples)) / totalSamples
plt.figure(figsize=(8, 6))
plt.xlabel('m(Omage)')
plt.ylabel('cumsum error rate') 
plt.plot(cifar10Omega[::1000], errorRate[::1000], marker= 'o', label= 'Cifar10')
plt.legend()
plt.show()

oodlabels = {'SVHN': svhnLabel, 'LSUN': lsunLabel, 'CIFAR100': cifar100Label}
for oodName in oodDatasetNames:
    #print ("Out-of-domain dataset {} is testing...".format(oodName))
    oodOutputs = torch.load('../Outputs/{}-{}-{}-Outputs'.format(trainType, inDomainDs, oodName))
    oodLabels = oodlabels[oodName]

    oodOmega = GetScores(oodOutputs, method= 'omega')
    oodMass = GetScores(oodOutputs, method= 'mass')
    idx = torch.argsort(oodOmega)
    oodMass = oodMass[idx]
    oodLabels = oodLabels[idx]
    oodPreds = torch.argmax(oodMass, dim = 1)
    rightClassfied = torch.cumsum(oodPreds == oodLabels, dim = 0)
    totalSamples = torch.cumsum(torch.ones(rightClassfied.shape), dim = 0)
    errorRate = (totalSamples - rightClassfied.type_as(totalSamples) )/ totalSamples
    plt.figure(figsize=(8, 6))
    plt.xlabel('m(Omage)')
    plt.ylabel('cumsum error rate') 
    plt.plot(oodOmega[::100], errorRate[::100], marker= 'o', label= oodName)
    plt.legend()
    plt.show()
        


