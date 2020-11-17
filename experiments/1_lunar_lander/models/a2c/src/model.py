import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 128):
        super(Model, self).__init__()

        self.device = "cpu"
        
        self.layers_features = [ 
            nn.Linear(input_shape[0], hidden_count),
            nn.ReLU()
        ]

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
                            
        torch.nn.init.xavier_uniform_(self.layers_features[0].weight)
        
        torch.nn.init.xavier_uniform_(self.layers_actor[0].weight)
        torch.nn.init.xavier_uniform_(self.layers_actor[2].weight)

        torch.nn.init.xavier_uniform_(self.layers_critic[0].weight)
        torch.nn.init.xavier_uniform_(self.layers_critic[2].weight)

        
        self.model_features = nn.Sequential(*self.layers_features)
        self.model_features.to(self.device)

        self.model_actor = nn.Sequential(*self.layers_actor)
        self.model_actor.to(self.device)

        self.model_critic = nn.Sequential(*self.layers_critic)
        self.model_critic.to(self.device)

        print(self.model_features)
        print(self.model_actor)
        print(self.model_critic)
       
    def forward(self, state):
        features = self.model_features(state)

        logits = self.model_actor(features)
        value  = self.model_critic(features)

        return logits, value


     
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
    
