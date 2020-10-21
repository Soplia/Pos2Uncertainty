import torch
import matplotlib.pyplot as plt 

upperBoder = 100
classNumber = 10
batchSize = 100
col = classNumber
row = batchSize

#evidence = torch.tensor([[0, 0, 0, 0],
#                                        [2, 2, 2, 2],
#                                        [4, 4, 0, 0],
#                                        [8, 0, 0, 0],
#                                        [1, 3, 3, 1]]).float()
#batchSize, classNumber = evidence.shape[0], evidence.shape[1]
evidence = torch.clamp(torch.randn(row, col), min= 0.0)
alpha = evidence + 1
alphaAcc = torch.sum(alpha, dim= 1)
evidence = alpha / torch.unsqueeze(alphaAcc, dim = 1)

posty = torch.empty((batchSize, 1))

for col in range(classNumber):
    # 如果evidence中包含0 该怎么？
    # 如果使用狄利克雷 
    frac = torch.clamp(torch.unsqueeze(evidence[:, col], dim= 1 ) / evidence, max= 1.0)
    ret = torch.sum(torch.unsqueeze(evidence[:, col], dim= 1 ) * frac, dim= 1, keepdim= True)
    posty = torch.cat((posty, ret), dim= 1)

posty = posty[:, 1:]
posty = torch.where(torch.isnan(posty), torch.full_like(posty, 0), posty)

prePosty = torch.argmax(posty, dim= 1)
preEvid = torch.argmax(evidence, dim = 1)

notEqual = prePosty != preEvid
for idx in range(batchSize):
    if notEqual[idx]:
        #print (evidence[idx], " max= {}, posty= {}, evid= {}".format(torch.max(evidence[idx]), evidence[idx][prePosty[idx]], evidence[idx][preEvid[idx]]))
        print (" max= {}, posty= {}, evid= {}".format(torch.max(evidence[idx]), evidence[idx][prePosty[idx]], evidence[idx][preEvid[idx]]))
print("percentage of prePosty != preEvid= {}".format(torch.sum(notEqual).item() / batchSize))

x = torch.arange(1, batchSize + 1)
plt.xlabel('ith-Sample')
plt.ylabel('Prediction')
plt.plot(x, prePosty, '-r', label='prePosty')
plt.plot(x, preEvid,'--g', label='preEvid')
plt.legend()
plt.show()
