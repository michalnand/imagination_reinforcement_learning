import torch
import torch.nn as nn

import sys
#sys.path.insert(0, '../../..')
sys.path.insert(0, '../../../../..')

import libs_layers


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, weight_init_gain = 1.0):
        super(ResidualBlock, self).__init__()

        
        self.conv0  = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act0   = nn.ReLU()
        self.conv1  = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act1   = nn.ReLU()
            
        torch.nn.init.xavier_uniform_(self.conv0.weight, gain=weight_init_gain)
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=weight_init_gain)


    def forward(self, x):
        y  = self.conv0(x)
        y  = self.act0(y)
        y  = self.conv1(y)
        y  = self.act1(y + x)
        
        return y

  
class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        self.outputs_count  = outputs_count

        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]    

        fc_inputs_count = 64*(input_width//16)*(input_height//16)
  
        self.layers = [ 
            nn.Conv2d(input_channels*2 + outputs_count, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            ResidualBlock(64),
            ResidualBlock(64),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            ResidualBlock(64), 
            ResidualBlock(64),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            Flatten(),

            nn.Linear(fc_inputs_count, 256),
            nn.ReLU(),                       
            nn.Linear(256, 1),
            nn.Softplus() 
        ] 

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print(self.model)
        print("\n\n")

    def forward(self, state, state_next, action):
        action_ = action.unsqueeze(1).unsqueeze(1).transpose(3, 1).repeat((1, 1, self.input_shape[1], self.input_shape[2])).to(self.device)

        model_x = torch.cat([state, state_next, action_], dim = 1)
        
        return self.model(model_x)


    def save(self, path):
        print("saving ", path)
        torch.save(self.model.state_dict(), path + "trained/entropy_model.pt")
       
    def load(self, path):
        print("loading ", path) 

        self.model.load_state_dict(torch.load(path + "trained/entropy_model.pt", map_location = self.device))
        self.model.eval() 
      



if __name__ == "__main__":
    batch_size = 8

    channels = 4
    height   = 96
    width    = 96

    actions_count = 9


    state        = torch.rand((batch_size, channels, height, width))
    state_next   = torch.rand((batch_size, channels, height, width))
    action       = torch.rand((batch_size, actions_count))
    
    model = Model((channels, height, width), actions_count)


    variance = model.forward(state, state_next, action)

    print(variance)




