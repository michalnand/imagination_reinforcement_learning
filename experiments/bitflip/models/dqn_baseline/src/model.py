import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../../../../..')

import libs_layers

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 128):
        super(Model, self).__init__()

        self.device = "cpu"
        
        self.layers = [ 
            Flatten(),

            nn.Linear(input_shape[1]*input_shape[0], hidden_count),
            nn.ReLU(),           
          
            libs_layers.NoisyLinearFull(hidden_count, outputs_count)
        ]

        torch.nn.init.xavier_uniform_(self.layers[1].weight)
        torch.nn.init.xavier_uniform_(self.layers[3].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_dqn")
        print(self.model)
        print("\n\n")
       

    def forward(self, state):
        return self.model(state)

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "trained/model_dqn.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "trained/model_dqn.pt", map_location = self.device))
        self.model.eval()  
    
