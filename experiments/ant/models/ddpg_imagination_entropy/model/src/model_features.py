import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../../../../..')

import libs_layers


class Model(torch.nn.Module):
    def __init__(self, input_shape, hidden_count = 256):
        super(Model, self).__init__()

        self.device = "cpu"

        self.features_shape = (hidden_count, )
        
        self.layers = [ 
            nn.Linear(input_shape[0], hidden_count),
            nn.ReLU()           
        ]

        torch.nn.init.xavier_uniform_(self.layers[0].weight)
       
        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_features")
        print(self.model)
        print("\n\n")
       

    def forward(self, state):
        return self.model(state)

     
    def save(self, path):
        torch.save(self.model.state_dict(), path + "trained/model_features.pt")

    def load(self, path):       
        self.model.load_state_dict(torch.load(path + "trained/model_features.pt", map_location = self.device))
        self.model.eval()  
    
