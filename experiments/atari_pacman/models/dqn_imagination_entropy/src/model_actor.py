import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../../../../..')

import libs_layers


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]    


        fc_inputs_count = input_channels*input_width*input_height
  

        self.layers_value = [
            Flatten(),
            nn.Linear(fc_inputs_count, 512),
            nn.ReLU(),                       
            nn.Linear(512, 1)    
        ]  

        self.layers_advantage = [
            Flatten(),
            libs_layers.NoisyLinear(fc_inputs_count, 512),
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

        print("model_actor")
        print(self.model_value)
        print(self.model_advantage)
        print("\n\n")

    def forward(self, features):
        value       = self.model_value(features)
        advantage   = self.model_advantage(features)

        result = value + advantage - advantage.mean(dim=1, keepdim=True)

        return result

    def save(self, path):
        torch.save(self.model_value.state_dict(), path + "trained/model_actor_value.pt")
        torch.save(self.model_advantage.state_dict(), path + "trained/model_actor_advantage.pt")

    def load(self, path):
        self.model_value.load_state_dict(torch.load(path + "trained/model_actor_value.pt", map_location = self.device))
        self.model_advantage.load_state_dict(torch.load(path + "trained/model_actor_advantage.pt", map_location = self.device))
        
        self.model_value.eval() 
        self.model_advantage.eval() 



if __name__ == "__main__":
    batch_size = 8

    channels = 128
    height   = 6
    width    = 6

    actions_count = 9


    state   = torch.rand((batch_size, channels, height, width))

    model = Model((channels, height, width), actions_count)


    q_values = model.forward(state)

    print(q_values.shape)