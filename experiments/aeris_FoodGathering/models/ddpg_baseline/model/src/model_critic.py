import torch
import torch.nn as nn

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

        self.layers_features = [ 
            nn.Conv1d(self.channels, kernels_count, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(kernels_count, kernels_count, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            Flatten()
        ]

        self.layers_output = [ 
            nn.Linear(fc_count + outputs_count, hidden_count),
            nn.ReLU(),            
            nn.Linear(hidden_count, 1)           
        ] 

        torch.nn.init.xavier_uniform_(self.layers_features[0].weight)
        torch.nn.init.xavier_uniform_(self.layers_features[2].weight)

        torch.nn.init.xavier_uniform_(self.layers_output[0].weight)
        torch.nn.init.uniform_(self.layers_output[2].weight, -0.003, 0.003)
 
        self.model_features = nn.Sequential(*self.layers_features) 
        self.model_features.to(self.device)

        self.model_output = nn.Sequential(*self.layers_output) 
        self.model_output.to(self.device)

        print("model_critic")
        print(self.model_features)
        print(self.model_output)
        print("\n\n")
       

    def forward(self, state, action):
        features = self.model_features(state)
        x        = torch.cat([features, action], dim = 1)
        
        return self.model_output(x)

    def save(self, path):
        print("saving to ", path)
        torch.save(self.model_features.state_dict(), path + "trained/model_critic_features.pt")
        torch.save(self.model_output.state_dict(), path + "trained/model_critic_output.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model_features.load_state_dict(torch.load(path + "trained/model_critic_features.pt", map_location = self.device))
        self.model_output.load_state_dict(torch.load(path + "trained/model_critic_output.pt", map_location = self.device))
        self.model_features.eval()  
        self.model_output.eval()  
