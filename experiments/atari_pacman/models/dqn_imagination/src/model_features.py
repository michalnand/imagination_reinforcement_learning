import torch
import torch.nn as nn

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

    def __init__(self, input_shape):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]    


        fc_inputs_count = 128*(input_width//16)*(input_height//16)

        self.features_shape = (128, input_height//16, input_width//16)
  
        self.layers_encoder = [ 
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            ResidualBlock(128),
            ResidualBlock(128),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            ResidualBlock(128), 
            ResidualBlock(128),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        ] 

        self.layers_decoder = [
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            nn.Conv2d(64, input_channels, kernel_size=1, stride=1, padding=0)
        ]

  
        for i in range(len(self.layers_encoder)):
            if hasattr(self.layers_encoder[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_encoder[i].weight)

        for i in range(len(self.layers_decoder)):
            if hasattr(self.layers_decoder[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_decoder[i].weight)

      
        self.model_encoder = nn.Sequential(*self.layers_encoder)
        self.model_encoder.to(self.device)

        self.model_decoder = nn.Sequential(*self.layers_decoder)
        self.model_decoder.to(self.device)

        print("model_features")
        print(self.model_encoder)
        print(self.model_decoder)
        print("\n\n")

    def forward(self, state):
        features    = self.model_encoder(state)

        state_predicted       = self.model_decoder(features)
       
        return features, state_predicted

    def save(self, path):
        print("saving ", path)

        torch.save(self.model_encoder.state_dict(), path + "trained/model_features_encoder.pt")
        torch.save(self.model_decoder.state_dict(), path + "trained/model_features_decoder.pt")

    def load(self, path):
        print("loading ", path) 

        self.model_encoder.load_state_dict(torch.load(path + "trained/model_features_encoder.pt", map_location = self.device))
        self.model_decoder.load_state_dict(torch.load(path + "trained/model_features_decoder.pt", map_location = self.device))
        
        self.model_encoder.eval() 
        self.model_decoder.eval() 

if __name__ == "__main__":
    batch_size = 8

    channels = 4
    height   = 96
    width    = 96

    actions_count = 9


    state   = torch.rand((batch_size, channels, height, width))

    model = Model((channels, height, width))

    features, state_predicted = model.forward(state)

    print(features.shape, state_predicted.shape)


