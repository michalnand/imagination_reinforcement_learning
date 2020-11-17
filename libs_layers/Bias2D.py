import torch
import torch.nn as nn
import numpy

class Bias2D(nn.Module):
    def __init__(self, channels, height, width):
        super(Bias2D, self).__init__()
        self.bias   = nn.Parameter(torch.zeros(channels, height, width))
        
    def forward(self, x): 
        return x + self.bias
