import torch
import torch.nn as nn


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
        
        input_channels  = self.input_shape[0]
        
        self.layers = [ 
            nn.Conv2d(input_channels + outputs_count, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128), 
            ResidualBlock(128),

            nn.Conv2d(128, input_channels, kernel_size=1, stride=1, padding=0)
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print(self.model)
        print("\n\n")


    def forward(self, state, action):
        action_ = action.unsqueeze(1).unsqueeze(1).transpose(3, 1).repeat((1, 1, self.input_shape[1], self.input_shape[2])).to(self.device)

        model_x = torch.cat([state, action_], dim=1)

        return self.model(model_x)
       

    def save(self, path):
        print("saving ", path)

        torch.save(self.model_forward.state_dict(), path + "trained/model_forward.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_forward.load_state_dict(torch.load(path + "trained/model_forward.pt", map_location = self.device))
        self.model_forward.eval() 
