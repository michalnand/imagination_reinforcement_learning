import torch
import torch.nn as nn


class ResidualBlock1d(torch.nn.Module):
    def __init__(self, channels, weight_init_gain = 1.0):
        super(ResidualBlock1d, self).__init__()

        
        self.conv0  = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act0   = nn.ReLU()
        self.conv1  = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
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
    def __init__(self, input_shape, outputs_count, kernels_count = 32):
        super(Model, self).__init__()

        self.device = "cpu"
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.channels = input_shape[0]
        
        self.layers = [ 
            nn.Conv1d(self.channels + outputs_count, kernels_count, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            
            ResidualBlock1d(kernels_count),
            ResidualBlock1d(kernels_count),

            nn.ConvTranspose1d(kernels_count, kernels_count, kernel_size=8, stride=4, padding=2, output_padding=0),
            nn.ReLU(),
            nn.Conv1d(kernels_count, self.channels, kernel_size=3, stride=1, padding=1)
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
        a_ = action.unsqueeze(2).repeat(1, 1, state.shape[2])

        x = torch.cat([state, a_], dim = 1) 
        
        return self.model(x) + state

    def save(self, path):
        torch.save(self.model.state_dict(), path + "trained/model_forward.pt")

    def load(self, path):       
        self.model.load_state_dict(torch.load(path + "trained/model_forward.pt", map_location = self.device))
        self.model.eval()  
    

if __name__ == "__main__":
    batch_size      = 1
    input_shape     = (6, 32)
    outputs_count   = 5

    model = Model(input_shape, outputs_count)

    state   = torch.randn((batch_size, ) + input_shape)
    action  = torch.randn((batch_size, outputs_count))

    y = model.forward(state, action)

    print(y.shape)
