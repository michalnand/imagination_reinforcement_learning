import torch
import torch.nn as nn
import numpy

class NoisyLinearFull(torch.nn.Module):
    def __init__(self, in_features, out_features, sigma = 1.0):
        super(NoisyLinearFull, self).__init__()
        
        self.out_features   = out_features
        self.in_features    = in_features
        self.sigma          = sigma

        self.weight  = nn.Parameter(torch.zeros(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias  = nn.Parameter(torch.zeros(out_features))
 

        self.weight_noise  = nn.Parameter(torch.zeros(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight_noise)

        self.bias_noise  = nn.Parameter((0.1/out_features)*torch.randn(out_features)) 
 

    def forward(self, x): 
        noise           = self.sigma*torch.randn((self.in_features, self.out_features)).to(x.device).detach()
        bias_noise      = self.sigma*torch.randn((self.out_features)).to(x.device).detach()

        weight_noised   = self.weight + self.weight_noise*noise
        bias_noised     = self.bias   + self.bias_noise*bias_noise 

        return x.matmul(weight_noised) + bias_noised

    def __repr__(self):
        return "NoisyLinearFull(in_features={}, out_features={}, sigma={})".format(self.in_features, self.out_features, self.sigma)
        

if __name__ == "__main__":
    in_features     = 32
    out_features    = 16

    layer = NoisyLinearFull(in_features, out_features)

    input  = torch.randn((10, in_features))
    
    for i in range(4):
        output = layer.forward(input)
        print(output, "\n\n\n")
    