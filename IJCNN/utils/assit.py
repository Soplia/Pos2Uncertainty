import torch
import torch.nn.functional as F

def GetDevice():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')    

def ToDevice(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [ToDevice(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield ToDevice(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
def L2Loss(inputs):
    return torch.sum(inputs ** 2) / 2
def relu_evidence(logits):
    return F.relu(logits)

def KL(alpha, numOfClass, device= GetDevice()):
    beta = torch.ones((1, numOfClass), dtype = torch.float32, requires_grad= False).to(device)
    #beta = prioriC * prioriProbability
    S_alpha = torch.sum(alpha, dim= 1, keepdims= True)
    S_beta = torch.sum(beta, dim = 1, keepdims= True)
    lnB = torch.lgamma(input= S_alpha) - torch.sum(torch.lgamma(alpha), dim= 1, keepdims= True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim= 1, keepdims= True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(input= S_alpha)
    dg1 = torch.digamma(input= alpha)
    
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim = 1, keepdims= True) + lnB + lnB_uni
    return kl

def loss_eq5(labels, alpha, numOfClass, global_step, annealing_step):
    S = torch.sum(alpha, dim = 1, keepdims = True)
    logLikeHood = torch.sum ((labels - (alpha / S)) ** 2, dim = 1, keepdims= True) + \
                              torch.sum (alpha * (S - alpha) / (S * S * (S + 1)), dim = 1, keepdims= True)
    KL_reg = min(1.0, float(global_step) / annealing_step) * \
                     KL((alpha - 1) * (1 - labels) + 1, numOfClass)
    #prioriC * prioriProbability
    return logLikeHood + KL_reg

