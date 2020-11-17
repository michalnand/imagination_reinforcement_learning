import torch
import torch.nn as nn
import numpy

class GRU(nn.Module):
    def __init__(self, in_features, out_features):

        super(GRU, self).__init__()

        in_count = in_features + out_features

        layers_gate = [
            nn.Linear(in_count, out_features, bias = True),
            nn.Sigmoid()
        ]

        layers_candidate = [
            nn.Linear(in_count, out_features, bias = True),
            nn.Tanh()
        ]

       
        torch.nn.init.xavier_uniform_(layers_gate[0].weight)
        torch.nn.init.constant_(layers_gate[0].bias, 4.0)
        torch.nn.init.xavier_uniform_(layers_candidate[0].weight)

        self.gate       = torch.nn.Sequential(*layers_gate)
        self.candidate  = torch.nn.Sequential(*layers_candidate)


    def forward(self, x, h): 

        x_  = torch.cat([x, h], dim = 1)

        gate        = self.gate(x_)
        candidate   = self.candidate(x_)

        h_new = (1.0 - gate)*h + gate*candidate

        return h_new


if __name__ == "__main__":
    batch_size      = 8
    in_features     = 100
    out_features    = 16

    
    model       = GRU(in_features, out_features)
    optimizer   = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        
        h = torch.zeros(batch_size, out_features)

        for t in range(32):
            x = torch.randn(batch_size, in_features)
            
            h_new = model.forward(x, h)
            h = h_new.clone()
        

        optimizer.zero_grad() 
        loss        = ((0.3 - h)**2).mean()
        loss.backward()
        optimizer.step()  

        print("loss = ", loss)
        print("h    = ", h)
        print("\n\n")
