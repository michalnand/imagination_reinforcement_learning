import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 128):
        super(Model, self).__init__()

        self.device = "cpu"

        state_size = input_shape[1]*input_shape[0]
        self.layers = [ 
            Flatten(),

            nn.Linear(state_size + outputs_count, hidden_count),
            nn.ReLU(),           
          
            nn.Linear(hidden_count, state_size)
        ]
 
        torch.nn.init.xavier_uniform_(self.layers[1].weight)
        torch.nn.init.xavier_uniform_(self.layers[3].weight)

        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model_forward")
        print(self.model)
        print("\n\n")
       

    def forward(self, state, action):
        s_ = state.view(state.size(0), -1)
        x = torch.cat([s_, action], dim=1)

        y = self.model(x)

        y = y.reshape(state.shape) + state

        return y

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "trained/model_forward.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "trained/model_forward.pt", map_location = self.device))
        self.model.eval()  
    
