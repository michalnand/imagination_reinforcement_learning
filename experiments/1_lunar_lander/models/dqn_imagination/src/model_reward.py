import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 256):
        super(Model, self).__init__()

        self.device = "cpu"
        
        self.layers = [ 
            nn.Linear(input_shape[0] + outputs_count, hidden_count),
            nn.ReLU(),              
            nn.Linear(hidden_count, 64),
            nn.ReLU(),              
            nn.Linear(64, 1)
        ]
 
        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_reward")
        print(self.model)
        print("\n\n")
       
    def forward(self, state, action):
        x = torch.cat([state, action], dim = 1)
        return self.model(x)

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "trained/model_reward.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "trained/model_reward.pt", map_location = self.device))
        self.model.eval()  
    
