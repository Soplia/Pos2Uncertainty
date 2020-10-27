# 使用"训练好的LeNet模型(使用CrossEntropyLoss、dirichlet损失函数，基于前五个类训练的)"
# 进行proErr, uncertainty

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt 
from utils.model import MnistNN
from utils.countNp import GetUncFre

#from matplotlib.pyplot import MultipleLocator

print ("Loading data set from ../../Dataset/mnist/MnistTest-AllTenClasses.npz")
testNp = np.load('../Dataset/mnist/MnistTest-AllTenClasses.npz')
testImageNp =  testNp['arr_0'] #[10000, 784]
testLabelNp =  testNp['arr_1'] #[10000, 10]
testLabelSingleNp = np.argmax(testLabelNp, axis= 1)
testImageTh = torch.from_numpy(testImageNp)
testLabelTh = torch.from_numpy(testLabelSingleNp).type(torch.LongTensor) 
print ("测试数据集中前五个类所占比重{}".format(torch.sum(testLabelTh < 5).item() / testLabelTh.shape[0]))

batchSize = 100
classNumber = 5
modelType = ['CNN', 'EDL', 'CNN-EDL'] 
lineStyle = {'CNN': '-.k', 'EDL': '-r', 'CnnEdl': '--b'}

testDS = torch.utils.data.TensorDataset(testImageTh, testLabelTh)
testLoader = torch.utils.data.DataLoader(testDS, batch_size = batchSize, shuffle = False)

model = MnistNN(classNumber)

for  modelName in modelType:
    if modelName == 'CNN-EDL':
        model.load_state_dict(torch.load('../DictState/Cifar10-CNN.pt'))
    else:
        model.load_state_dict(torch.load('../DictState/Cifar10-{}.pt'.format(modelName)))
    print("{} Testing ...".format(modelName))
    model.eval()
    accNum = 0.0
    uList = []
    predLabeList = []
    
    for j, (images, labels) in enumerate(testLoader):
        testDS = images.view(batchSize, 1, 28, 28)
        outputs = model(testDS) 

        if modelName == 'CNN':
            evidence = F.softmax(outputs, dim= 1)
            proErr = 1 - torch.max(evidence, dim= 1).values
            uList.append(proErr)
            predLabel = torch.squeeze(torch.argmax(evidence, dim= 1))
            predLabeList.append(predLabel)
            accNum += torch.sum(torch.argmax(evidence, dim= 1) == labels).item()
        elif modelName == 'CnnEdl':
            evidence = F.relu(outputs)
            alpha = evidence + 1
            alphaAcc = torch.sum (alpha, dim = 1, keepdims = True)
            prioriP = alpha / alphaAcc
            u = torch.squeeze(classNumber / alphaAcc) 
            uList.append(u)
            predLabel = torch.squeeze(torch.argmax(prioriP, dim= 1))
            predLabeList.append(predLabel)
            accNum += torch.sum(torch.argmax(prioriP, dim= 1) == labels).item()
        else:
            evidence = F.relu(outputs)
            alpha = evidence + 1
            alphaAcc = torch.sum (alpha, dim = 1, keepdims = True)
            prioriP = alpha / alphaAcc
            u = torch.squeeze(classNumber / alphaAcc) 
            uList.append(u)
            predLabel = torch.squeeze(torch.argmax(prioriP, dim= 1))
            predLabeList.append(predLabel)
            accNum += torch.sum(torch.argmax(prioriP, dim= 1) == labels).item()

    acc =  accNum / testLabelTh.shape[0]
    print("{} Test acc= {}".format(modelName, acc)) 

    uncer = torch.cat(uList, 0)
    uncKeyFre = GetUncFre(uncer)

    plt.xlabel('Uncer')
    plt.ylabel('Freq')
    plt.plot(list(i / uncKeyFre['decNum'] for i  in uncKeyFre['key']) , list(uncKeyFre['fre']), lineStyle[modelName], label= modelName)
    plt.legend()
plt.show()

