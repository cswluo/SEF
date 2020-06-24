import torch
import torch.nn as nn
import pdb

class SoftSigmoid(nn.Module):

    def __init__(self):
        super(SoftSigmoid, self).__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.bias = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
    
        sval = torch.reciprocal(1+torch.exp(torch.neg(torch.mul(x, self.weight)+self.bias)))

        return sval

def SoftCrossEntropy(p, q):

    assert p.shape == q.shape, 'the size of p and q must be euqal.'

    nsamples = p.shape[0]

    loss = torch.div(torch.sum(torch.mul(p, torch.neg(torch.log(q)))), nsamples)

    return loss
