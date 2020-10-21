import torch
import matplotlib.pyplot as plt 

def GetUncFre(tsr):
    key = torch.unique(tsr)
    fre = dict()
    for k in key:
        mask = (tsr == k)
        tsrNew = tsr[mask]
        v = tsrNew.shape[0]
        fre[k.item()] = v
    freSorted = dict(sorted(fre.items(), key= lambda item: item[0]))
    return {'key': freSorted.keys(), 'fre': freSorted.values()}

batchSize = 100
row = batchSize
decNum = 10
evidence = torch.round(torch.randn(row) * decNum)

uncKeyFre = GetUncFre(evidence)

plt.xlabel('Uncer')
plt.ylabel('Freq')
plt.plot(list(i / decNum for i  in uncKeyFre['key']) , list(uncKeyFre['fre']), '-r', label='uncer')
plt.legend()
plt.show()


