import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../../../../..')

import libs_layers

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, kernels_count = 32, hidden_count = 256):
        super(Model, self).__init__()

        #self.device = "cpu"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.channels   = input_shape[0]
        self.width      = input_shape[1]

        fc_count        = kernels_count*self.width//4

        self.layers = [ 
            nn.Conv1d(self.channels, kernels_count, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(kernels_count, kernels_count, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            Flatten(),

            libs_layers.NoisyLinearFull(fc_count, hidden_count),
            nn.ReLU(),            
            libs_layers.NoisyLinearFull(hidden_count, outputs_count),
            nn.Tanh()
        ]

        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)
        torch.nn.init.xavier_uniform_(self.layers[5].weight)
        torch.nn.init.uniform_(self.layers[7].weight, -0.3, 0.3)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_actor")
        print(self.model)
        print("\n\n")
       

    def forward(self, state):
        return self.model(state)

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "trained/model_actor.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "trained/model_actor.pt", map_location = self.device))
        self.model.eval()  
    
