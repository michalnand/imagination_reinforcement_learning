import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../../../../..')

import libs_layers

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 256):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.robots_count    = input_shape[0]
        self.features_count  = input_shape[1]
        self.actions_count   = outputs_count//self.robots_count

        self.layers = [  
            nn.Linear(self.features_count, hidden_count),
            nn.ReLU(),            
            
            libs_layers.NoisyLinearFull(hidden_count, hidden_count//2),
            nn.ReLU(),   

            libs_layers.NoisyLinearFull(hidden_count//2, self.actions_count),
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
        batch_size      = state.shape[0]

        state_  = state.reshape((batch_size*self.robots_count, self.features_count))
        y_      = self.model(state_) 
        y       = y_.reshape((batch_size, self.robots_count*self.actions_count))

        return y

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "trained/model_actor.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "trained/model_actor.pt", map_location = self.device))
        self.model.eval()  
    
if __name__ == "__main__":
    batch_size      = 32
    robots_count    = 64
    features_count  = 26

    input_shape     = (robots_count, features_count)
    outputs_count   = 5*robots_count

    model = Model(input_shape, outputs_count)

    state   = torch.randn((batch_size, robots_count, features_count))

    y = model.forward(state)

    print(y.shape)
