import torch
import torch.nn as nn


class ResidualBlock1D(torch.nn.Module):
    def __init__(self, channels, weight_init_gain = 1.0):
        super(ResidualBlock1D, self).__init__()

        self.conv0 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act0  = nn.ReLU()

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act1  = nn.ReLU()

        torch.nn.init.xavier_uniform_(self.conv0.weight, gain=weight_init_gain)
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=weight_init_gain)

    def forward(self, x):
        y = self.conv0(x)
        y = self.act0(y)

        y = self.conv1(y)
        y = self.act1(y + x)
        
        return y


class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, kernels_count = 128):
        super(Model, self).__init__()

        self.device = "cpu"
        
        self.layers = [ 
            nn.Conv1d(input_shape[0] + outputs_count, out_channels=kernels_count, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),

            ResidualBlock1D(kernels_count),
            ResidualBlock1D(kernels_count),
            ResidualBlock1D(kernels_count),

            nn.Conv1d(kernels_count, out_channels=input_shape[0], kernel_size=1, stride = 1, padding=0)
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)


        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_forward")
        print(self.model)
        print("\n\n")
       
    def forward(self, state, action):
        x = torch.cat([state, action], dim = 1) 
        y = self.model(x)
        return y

    def save(self, path):
        torch.save(self.model.state_dict(), path + "trained/model_forward.pt")

    def load(self, path):       
        self.model.load_state_dict(torch.load(path + "trained/model_forward.pt", map_location = self.device))
        self.model.eval()  
    
if __name__ == "__main__":

    features_count  = 20
    sequence_length = 64

    outputs_count   = 5
    batch_size      = 1

    model = Model((features_count, sequence_length), outputs_count)

    state   = torch.randn((batch_size, ) + (features_count, sequence_length))
    action  = torch.randn((batch_size, ) + (outputs_count, sequence_length))

    y = model(state, action)

    print(">>> ", y.shape)