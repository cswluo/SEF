import os, sys
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets

# curpath = os.getcwd()
progpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(progpath))
from pymodels import resnet
modelpath = os.path.join(os.path.dirname(progpath),'pymodels')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DistLoss(nn.Module):

    def __init__(self, rho=0, nparts=1):
        self.rho = rho
        self.nparts = nparts

    def forward(self, x):
        # x: n_features * nparts
        x_norm = torch.norm(x, dim=0)
        x_norm_matrix = torch.ger(x_norm, x_norm)

        x_inner = torch.mm(x.T, x)
        self_similarity = torch.div(x_inner, x_norm_matrix)
        n_pairs = (torch.numel(self_similarity) - self.nparts)/2
        loss = ((torch.sum(self_similarity) - torch.diagonal(self_similarity).sum())/2/n_pairs).to(device)

        return loss


if __name__ == '__main__':

    model = resnet.resnet18(pretrained=True, model_dir=modelpath)
    nparts = 3
    dl = DistLoss(0.1, nparts=nparts)
    x = torch.rand(5,nparts, requires_grad=True)
    loss = dl.forward(x)
    loss.backward()

    print(loss, x.grad)