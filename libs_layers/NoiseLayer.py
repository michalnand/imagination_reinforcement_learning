import torch
import torch.nn as nn
import numpy

class NoiseLayer(torch.nn.Module):
    def __init__(self, inputs_count, sigma = 0.01):
        super(NoiseLayer, self).__init__()
        
        self.inputs_count   = inputs_count
        self.sigma          = sigma
        self.w              = torch.nn.Parameter(torch.randn(self.inputs_count)) 
 
    def forward(self, x):
        noise = self.sigma*torch.randn((x.shape[0], self.inputs_count)).to(x.device)
        return x + self.w*noise 


if __name__ == "__main__":
    in_features     = 1024
    out_features    = 5


    layer = NoisyLinear(in_features, out_features)

    for j in range(4):
        input  = torch.randn(in_features).unsqueeze(0)
        for i in range(10):
            output = layer.forward(input)
            print(output)
        print("\n\n\n")
    