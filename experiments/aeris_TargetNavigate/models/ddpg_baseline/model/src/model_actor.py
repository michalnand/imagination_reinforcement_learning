import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../../../../..')

import libs_layers

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 512):
        super(Model, self).__init__()

        self.device = "cpu"
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.channels   = input_shape[0]
        self.width      = input_shape[1]

        self.layers = [ 
            nn.Linear(self.channels*self.width, hidden_count),
            nn.ReLU(),

            libs_layers.NoisyLinearFull(hidden_count, hidden_count//2),
            nn.ReLU(),

            libs_layers.NoisyLinearFull(hidden_count//2, outputs_count),
            nn.Tanh()
        ]

        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)
        torch.nn.init.uniform_(self.layers[4].weight, -0.3, 0.3)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_actor")
        print(self.model)
        print("\n\n")
       

    def forward(self, state):
        x = state.view(state.size(0), -1)

        return self.model(x)

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "trained/model_actor.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "trained/model_actor.pt", map_location = self.device))
        self.model.eval()  
    
if __name__ == "__main__":
    batch_size      = 1
    input_shape     = (6, 32)
    outputs_count   = 5

    model = Model(input_shape, outputs_count)

    state   = torch.randn((batch_size, ) + input_shape)

    y = model.forward(state)

    print(y.shape)
