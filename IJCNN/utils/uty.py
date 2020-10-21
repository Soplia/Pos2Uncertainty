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

#evidence = torch.randint(low= 0, high= upperBoder, size= (row, col)).float()
evidence = torch.randn(row, col)
alpha = evidence + 1
alphaAcc = torch.sum(alpha, dim= 1)
K = classNumber
uDich =  K / alphaAcc

alphaVar = torch.var(alpha, dim= 1)
alphaVar = torch.where(alphaVar <= 0.0, torch.full_like(alphaVar, 1), alphaVar)
uVar = K / alphaVar

uVarDich = uVar + uDich
uVarRatio = uVar / uVarDich
uDichRatio = uDich / uVarDich
uMer = uVarRatio * uVar + uDichRatio * uDich
#uMer = uVarRatio * uDich + uDichRatio * uVar

x = torch.arange(1, batchSize + 1)
plt.xlabel('ith-Sample')
plt.ylabel('Uncertainty')
plt.plot(x, uVar, '-r', label='uVar')
plt.plot(x, uDich,'--g', label='uDich')
plt.plot(x, uMer, ':b', label='uMer')
plt.legend()
plt.show()
