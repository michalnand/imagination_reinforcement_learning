import torch
import torch.nn as nn
import numpy

class LSTM(nn.Module):
    def __init__(self, in_features, out_features):

        super(LSTM, self).__init__()

        in_count = in_features + out_features

        self.input_gate = torch.nn.Sequential(
            nn.Linear(in_count, out_features, bias = True),
            nn.Sigmoid()
        )

        self.udpate_gate = torch.nn.Sequential(
            nn.Linear(in_count, out_features, bias = True),
            nn.Sigmoid()
        )

        self.candidate_gate = torch.nn.Sequential(
            nn.Linear(in_count, out_features, bias = True),
            nn.Tanh()
        )

        self.output_gate = torch.nn.Sequential(
            nn.Linear(in_count, out_features, bias = True),
            nn.Sigmoid()
        )

    def forward(self, x, h, c): 

        x_  = torch.cat([x, h], dim = 1)

        f   = self.input_gate(x_)
        u   = self.udpate_gate(x_)
        cc  = self.candidate_gate(x_)
        o   = self.output_gate(x_)

        c_new  = f*c + u*cc
        h_new  = torch.tanh(c_new)*o

        return h_new, c_new


if __name__ == "__main__":
    batch_size      = 8
    in_features     = 100
    out_features    = 16

    
    

    model       = LSTM(in_features, out_features)
    optimizer   = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        
        h = torch.zeros(batch_size, out_features)
        c = torch.zeros(batch_size, out_features)

        for t in range(32):
            x = torch.randn(batch_size, in_features)
            
            h_new, c_new = model.forward(x, h, c)
            h = h_new.clone()
            c = c_new.clone()


        optimizer.zero_grad() 
        loss        = ((0.3 - h)**2).mean()
        loss.backward()
        optimizer.step()  

        print("loss = ", loss)
        print("h    = ", h)
        print("\n\n")
