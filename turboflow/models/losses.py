import torch
import torch.nn as nn
import torch.nn.functional as F


class NSloss(nn.Module):
    '''
    Loss based Navier-Stokes equation
    '''

    def __init__(self, p=2, average=True):
        super(NSloss, self).__init__()
        self.avg = average
        self.p = p

    def forward(self, inputs, targets):
        
        return  inputs