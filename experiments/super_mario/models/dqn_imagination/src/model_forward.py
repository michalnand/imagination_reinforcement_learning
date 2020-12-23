import torch
import torch.nn as nn

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, weight_init_gain = 1.0):
        super(ResidualBlock, self).__init__()

        self.conv0      = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act0       = nn.ReLU()
        self.conv1      = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act1       = nn.ReLU()
            
        torch.nn.init.xavier_uniform_(self.conv0.weight, gain=weight_init_gain)
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=weight_init_gain)

    def forward(self, x):
        y  = self.conv0(x)
        y  = self.act0(y)
        y  = self.conv1(y)
        y  = self.act1(y + x)
        
        return y

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, kernels_count = 64):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = [
            nn.Conv2d(input_shape[0] + outputs_count, kernels_count, kernel_size=8, stride=8, padding=0),
            nn.ReLU(),

            ResidualBlock(kernels_count),
            ResidualBlock(kernels_count),

            nn.ConvTranspose2d(kernels_count, kernels_count, kernel_size=8, stride=8, padding=0),
            nn.ReLU(),
            nn.Conv2d(kernels_count, input_shape[0], kernel_size=3, stride=1, padding=1)
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
        height  = state.shape[2]
        width   = state.shape[3]
        action_ = action.unsqueeze(2).unsqueeze(2).repeat(1, 1, height, width)

        x = torch.cat([state, action_], dim=1)

        return self.model(x) + state

    def save(self, path):
        torch.save(self.model.state_dict(), path + "trained/model_forward.pt")
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path + "trained/model_forward.pt", map_location = self.device))
        self.model.eval() 

if __name__ == "__main__":
    batch_size = 8

    channels = 3
    height   = 96
    width    = 96

    actions_count = 9


    state           = torch.rand((batch_size, channels, height, width))
    action          = torch.rand((batch_size, actions_count))

    model = Model((channels, height, width), actions_count)

    state_predicted = model.forward(state, action)

    print(state_predicted.shape)
