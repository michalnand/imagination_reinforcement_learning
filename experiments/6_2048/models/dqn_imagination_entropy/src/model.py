import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, features_count = 256, hidden_count = 256):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.input_shape    = input_shape
        self.outputs_count  = outputs_count 
        
        self.layers = []

        self.layers.append(nn.Conv2d(input_shape[0], features_count, kernel_size=2, stride=1, padding=0))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.Conv2d(features_count, features_count, kernel_size=2, stride=1, padding=0))
        self.layers.append(nn.ReLU())

        self.layers.append(Flatten())

        self.layers.append(nn.Linear(features_count*2*2, hidden_count))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_count, outputs_count))

      
        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)


        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print(self.model)
      
    def forward(self, state):
        return self.model(state)

    def save(self, path):
        name = path + "trained/model_dqn.pt"
        print("saving", name)
        torch.save(self.model.state_dict(), name)

    def load(self, path):
        name = path + "trained/model_dqn.pt"
        print("loading", name) 

        self.model.load_state_dict(torch.load(name, map_location = self.device))
        self.model.eval() 
