import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../../..')

import libs_layers

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 128):
        super(Model, self).__init__()

        self.device = "cpu"

        self.outputs_count  = outputs_count
        self.hidden_count   = hidden_count
        
        '''
        self.layers_features  = [
            nn.Linear(input_shape[0] + hidden_count, hidden_count),
            nn.ReLU()    
        ]
        '''
        self.model_features = libs_layers.GRU(input_shape[0], hidden_count)
        
        self.layers_actor  = [
            nn.Linear(hidden_count, hidden_count//2),
            nn.ReLU(),    
            nn.Linear(hidden_count//2, outputs_count)
        ]

        self.layers_critic  = [
            nn.Linear(hidden_count, hidden_count//2),
            nn.ReLU(),    
            nn.Linear(hidden_count//2, 1)
        ]           


        torch.nn.init.xavier_uniform_(self.layers_actor[0].weight)
        torch.nn.init.xavier_uniform_(self.layers_actor[2].weight)

        torch.nn.init.xavier_uniform_(self.layers_critic[0].weight)
        torch.nn.init.xavier_uniform_(self.layers_critic[2].weight)

        
        #self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_actor = nn.Sequential(*self.layers_actor)
        self.model_actor.to(self.device)

        self.model_critic = nn.Sequential(*self.layers_critic)
        self.model_critic.to(self.device)

        self.reset()

        print(self.model_features)
        print(self.model_actor)
        print(self.model_critic)
       
    def forward(self, state):
        hidden_state = self.model_features(state, self.hidden_state)

        logits = self.model_actor(hidden_state)  
        value  = self.model_critic(hidden_state)

        self.hidden_state = hidden_state.clone()

        return logits, value

    def reset(self):
        self.hidden_state   = torch.zeros((1, self.hidden_count)).to(self.device)
        
    def save(self, path):
        print("saving to ", path)

        torch.save(self.model_features.state_dict(), path + "trained/model_features.pt")
        torch.save(self.model_actor.state_dict(), path + "trained/model_actor.pt")
        torch.save(self.model_critic.state_dict(), path + "trained/model_critic.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model_features.load_state_dict(torch.load(path + "trained/model_features.pt", map_location = self.device))
        self.model_actor.load_state_dict(torch.load(path + "trained/model_actor.pt", map_location = self.device))
        self.model_critic.load_state_dict(torch.load(path + "trained/model_critic.pt", map_location = self.device))

        self.model_features.eval() 
        self.model_actor.eval() 
        self.model_critic.eval()  
    
