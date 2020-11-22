import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../../../../..')

import libs_layers


class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 128):
        super(Model, self).__init__()

        self.device = "cpu"
        
        self.layers = [      
            libs_layers.NoisyLinear(input_shape[0], hidden_count),
            nn.ReLU(),    
            libs_layers.NoisyLinear(hidden_count, outputs_count),
            nn.Tanh()
        ]

        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.uniform_(self.layers[2].weight, -0.3, 0.3)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_actor")
        print(self.model)
        print("\n\n")
       

    def forward(self, state):
        return self.model(state)

     
    def save(self, path):
        torch.save(self.model.state_dict(), path + "trained/model_actor.pt")

    def load(self, path):       
        self.model.load_state_dict(torch.load(path + "trained/model_actor.pt", map_location = self.device))
        self.model.eval()  
    
