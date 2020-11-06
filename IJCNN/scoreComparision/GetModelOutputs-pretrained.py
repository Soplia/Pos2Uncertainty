# 使用基于CIFAR10 + CEL 训练好的模型，
# 获得对于不同数据集的模型输出，以避免每次重复计算模型输出
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import matplotlib.pyplot as plt 
from utils.model import WideResNet
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
testLoaderCifar10 = torch.utils.data.DataLoader(testsetCifar, batch_size=batchSize, shuffle=True, num_workers=0)

print ("Loading LSUN dataset...")
# lsunFeature Size([3000, 3, 32, 32])
lsunFeature = torch.load('../Dataset/lsun/LSUNpicFeature').type(torch.FloatTensor)
#  lsunOnehotLabel Size([3000, 10])
lsunOnehotLabel = torch.load('../Dataset/lsun/LSUNpicLabel').type(torch.FloatTensor)
# lsunLabel Size([3000])
lsunLabel = torch.squeeze(torch.argmax(lsunOnehotLabel, dim = 1))
lsunTestset = torch.utils.data.TensorDataset(lsunFeature, lsunLabel)
testLoaderLsun = torch.utils.data.DataLoader(lsunTestset, batch_size=batchSize, shuffle=True, num_workers=0)

print ("Loading Cifar100 Test Dataset...")
# lsunFeature Size([10000, 3, 32, 32])
cifar100Feature = torch.load('../Dataset/cifar100/cifar100-test-features').type(torch.FloatTensor)
# lsunLabel Size([10000])
cifar100Label = torch.load('../Dataset/cifar100/cifar100-test-labels')
cifar100Testset = torch.utils.data.TensorDataset(cifar100Feature, cifar100Label)
testLoaderCifar100 = torch.utils.data.DataLoader(cifar100Testset, batch_size=batchSize, shuffle=True, num_workers=0)

#print ("Loading KMNIST Test Dataset...")
## lsunFeature Size([10000, 3, 32, 32])
#kmnistFeature = torch.load('../Dataset/kmnist/kmnist-test-features').type(torch.FloatTensor)
## lsunLabel Size([10000])
#kmnistLabel = torch.load('../Dataset/kmnist/kmnist-test-labels')
#kmnistTestset = torch.utils.data.TensorDataset(kmnistFeature, kmnistLabel)
#testLoaderKmnist = torch.utils.data.DataLoader(kmnistTestset, batch_size=batchSize, shuffle=True, num_workers=0)

#print ("Loading FMNIST Test Dataset...")
## lsunFeature Size([10000, 3, 32, 32])
#fmnistFeature = torch.load('../Dataset/fmnist/fmnist-test-features').type(torch.FloatTensor)
## lsunLabel Size([10000])
#fmnistLabel = torch.load('../Dataset/fmnist/fmnist-test-labels')
#fmnistTestset = torch.utils.data.TensorDataset(fmnistFeature, fmnistLabel)
#testLoaderFmnist = torch.utils.data.DataLoader(fmnistTestset, batch_size=batchSize, shuffle=True, num_workers=0)

print ("Loading TEXTURE Test Dataset...")
# lsunFeature Size([10000, 3, 32, 32])
textureFeature = torch.load('../Dataset/texture/texture-test-features').type(torch.FloatTensor)[0:1300]
textureTestset = torch.utils.data.TensorDataset(textureFeature)
testLoaderTexture = torch.utils.data.DataLoader(textureTestset, batch_size=batchSize, shuffle=True, num_workers=0)

print ("Loading Places365 Test Dataset...")
# lsunFeature Size([10000, 3, 32, 32])
places365Feature = torch.load('../Dataset/places365/place365-test-features').type(torch.FloatTensor)
places365Testset = torch.utils.data.TensorDataset(places365Feature)
testLoaderplaces365 = torch.utils.data.DataLoader(places365Testset, batch_size=batchSize, shuffle=True, num_workers=0)

modelName = 'WideResNet' 
datasetNames = ['CIFAR10', 'SVHN'
                , 'CIFAR100', 'LSUN',
                'TEXTURE', 'PLACE365'
                ] #  'KMNIST', 'FMNIST',
testLoaders = {'CIFAR10': testLoaderCifar10, 'SVHN': testLoaderSvhn, 'LSUN': testLoaderLsun, 
               'CIFAR100': testLoaderCifar100, 'TEXTURE': testLoaderTexture, 'PLACE365': testLoaderplaces365}
#'KMNIST': testLoaderKmnist, 'FMNIST': testLoaderFmnist, 
device = GetDevice()
droprate = 0.3
classNumber = 10
totalNumLayers = 40
widenFactor = 2
model = WideResNet(totalNumLayers, classNumber, widenFactor, dropRate= droprate)
model.load_state_dict(torch.load('../DictState/pretrained/cifar10_wrn_pretrained_epoch_99.pt'))
ToDevice(model, device)
model.eval()

for datasetName in datasetNames:
    print("T= {}, {} Testing, model trained by {}...".format(datasetName, modelName))
    testLoader = DeviceDataLoader(testLoaders[datasetName], device)

    outputList = []
    if datasetName != 'TEXTURE' and datasetName != 'PLACE365':
        for j, (images, labels) in enumerate(testLoader):
            outputs = model(images)
            outputList.append(outputs.cpu().detach())
    else:
        for j, (images) in enumerate(testLoader):
            outputs = model(images[0])
            outputList.append(outputs.cpu().detach())
    outputTensor = torch.cat(outputList, 0)
    print ('outputTensor.shape: {} is saved!'.format(outputTensor.shape))
    torch.save(outputTensor, '../Outputs/{}-{}-Cifar10CNN-Outputs'.format(modelName, datasetName))




