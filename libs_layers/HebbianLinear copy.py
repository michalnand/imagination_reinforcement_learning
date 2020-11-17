import torch
import torch.nn as nn
import numpy

from torchviz import make_dot


class HebbianLinear(nn.Module):
    def __init__(self, in_features, out_features, device = "cpu", learning_rate = 0.001):
        super(HebbianLinear, self).__init__()

        self.in_features    = in_features
        self.out_features   = out_features
        self.device         = device
        self.learning_rate  = learning_rate


        #parameters - generalised hebbian rule, trained via backprop
        self.wa  = nn.Parameter(torch.zeros(out_features, in_features))
        self.wb  = nn.Parameter(torch.zeros(out_features, in_features))
        self.wc  = nn.Parameter(torch.zeros(out_features, in_features))
        self.wd  = nn.Parameter(torch.zeros(out_features, in_features))

        #xavier init
        r = numpy.sqrt(1.0/(self.in_features + self.out_features))
        self.wa.data.uniform_(-r, r)
        self.wb.data.uniform_(-r, r)
        self.wc.data.uniform_(-r, r)
        self.wd.data.uniform_(-r, r)

        #weights are common tensor, no parameter - grads will be computed via autograd, but values are no changed by optimizer
        self.weight = torch.zeros(out_features, in_features, requires_grad=True)
        self.reset()
        r = numpy.sqrt(1.0/(self.in_features + self.out_features))
        self.weight.data.uniform_(-r, r)


        #zero bias init, trained via backprop
        self.bias   = nn.Parameter(torch.zeros(out_features))
    
    def reset(self):
        r = numpy.sqrt(1.0/(self.in_features + self.out_features))
        rnd = r*(torch.rand(self.out_features, self.in_features) - 0.5)*2.0
        self.weight.data+= rnd


    def forward(self, x):
        #temporaty output
        x_  = x.detach()
        y_  = x_.mm(self.weight.t()).detach()
        
        #compute generalised hebbian rule : dw = X*Wa + Y*Wb + X*Y*Wc + Wd
        dw = 0

        #TODO - oh please, remove this batch for loop
        batch_size = x.shape[0]
        for b in range(x.shape[0]):
            dw+= (x_[b]*self.wa)/batch_size
            dw+= ((y_[b].t()*self.wb.t()).t())/batch_size
          
        dw+= (y_.t().mm(x_)*self.wc)/batch_size
        
        dw+= self.wd
        
        #update weights
        #self.weight = self.weight.data + self.learning_rate*dw
        eta = self.learning_rate
        self.weight = (1.0 - eta)*self.weight.data + eta*dw

        #compute output
        y = x.mm(self.weight.t()) + self.bias

        return y


if __name__ == "__main__":
    batch_size      = 1
    in_features     = 50
    out_features    = 10

    model = nn.Sequential(
        HebbianLinear(in_features, 256),
        nn.ReLU(),
        HebbianLinear(256, 64),
        nn.ReLU(),
        HebbianLinear(64, out_features))


    for i in range(10):
        input  = torch.randn(batch_size, in_features)
        output = model.forward(input)
        
        print(output)

        loss = output.mean()
        loss.backward()

    
    make_dot(loss).render("hebbian_linear_layer", format="png")
