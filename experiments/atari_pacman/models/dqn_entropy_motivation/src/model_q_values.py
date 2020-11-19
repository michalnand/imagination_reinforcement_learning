import torch
import torch.nn as nn

import sys
#sys.path.insert(0, '../../..')
sys.path.insert(0, '../../../../..')

import libs_layers


class Model(torch.nn.Module):

    def __init__(self, inputs_count, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers_value = [
            nn.Linear(inputs_count, 512),
            nn.ReLU(),                       
            nn.Linear(512, 1)    
        ]  

        self.layers_advantage = [
            libs_layers.NoisyLinear(inputs_count, 512),
            nn.ReLU(),                      
            libs_layers.NoisyLinear(512, outputs_count)
        ]
 
        for i in range(len(self.layers_value)):
            if hasattr(self.layers_value[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_value[i].weight)

        for i in range(len(self.layers_advantage)):
            if hasattr(self.layers_advantage[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_advantage[i].weight)

        self.model_value = nn.Sequential(*self.layers_value)
        self.model_value.to(self.device)

        self.model_advantage = nn.Sequential(*self.layers_advantage)
        self.model_advantage.to(self.device)

        print(self.model_value)
        print(self.model_advantage)
        print("\n\n")


    def forward(self, state):
        value       = self.model_value(state)
        advantage   = self.model_advantage(state)

        result = value + advantage - advantage.mean(dim=1, keepdim=True)

        return result

    def save(self, path):
        print("saving ", path)

        torch.save(self.model_value.state_dict(), path + "trained/model_value.pt")
        torch.save(self.model_advantage.state_dict(), path + "trained/model_advantage.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_value.load_state_dict(torch.load(path + "trained/model_value.pt", map_location = self.device))
        self.model_advantage.load_state_dict(torch.load(path + "trained/model_advantage.pt", map_location = self.device))
        
        self.model_value.eval() 
        self.model_advantage.eval() 
