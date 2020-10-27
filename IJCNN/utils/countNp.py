import torch
import matplotlib.pyplot as plt 

def GetUncFre(tsr, decNum = 1000):
    tsr = torch.round(tsr * decNum)
    key = torch.unique(tsr)
    fre = dict()
    for k in key:
        mask = (tsr == k)
        tsrNew = tsr[mask]
        v = tsrNew.shape[0]
        fre[k.item()] = v
    freSorted = dict(sorted(fre.items(), key= lambda item: item[0]))
    return {'key': freSorted.keys(), 'fre': freSorted.values(), 'decNum': decNum}

#batchSize = 100
#row = batchSize
#evidence = torch.randn(row)
#uncKeyFre = GetUncFre(evidence)
#plt.xlabel('Uncer')
#plt.ylabel('Freq')
#plt.plot(list(i / uncKeyFre['decNum'] for i  in uncKeyFre['key']) , list(uncKeyFre['fre']), '-r', label='uncer')
#plt.legend()
#plt.show()


