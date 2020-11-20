import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, hidden_count = 16):
        super(Model, self).__init__()

        self.device = "cpu"

        self.features_shape = (hidden_count//2, )

        self.layers_encoder = [ 
            nn.Linear(input_shape[0], hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, hidden_count//2),
            nn.ReLU()            
        ] 

        self.layers_decoder = [ 
            nn.Linear(hidden_count//2, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, input_shape[0])
        ]
 
        for i in range(len(self.layers_encoder)):
            if hasattr(self.layers_encoder[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_encoder[i].weight)

        for i in range(len(self.layers_decoder)):
            if hasattr(self.layers_decoder[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_decoder[i].weight)

      
        self.model_encoder = nn.Sequential(*self.layers_encoder) 
        self.model_decoder = nn.Sequential(*self.layers_decoder) 
        
        self.model_encoder.to(self.device)
        self.model_decoder.to(self.device)

        print("model_features")
        print(self.model_encoder)
        print(self.model_decoder)
        print("\n\n")
       

    def forward(self, state):
        features        = self.model_encoder(state)
        state_predicted = self.model_decoder(features)
        
        return features, state_predicted

     
    def save(self, path):
        print("saving to ", path)
        torch.save(self.model_encoder.state_dict(), path + "trained/model_features_encoder.pt")
        torch.save(self.model_decoder.state_dict(), path + "trained/model_features_decoder.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model_encoder.load_state_dict(torch.load(path + "trained/model_features_encoder.pt", map_location = self.device))
        self.model_decoder.load_state_dict(torch.load(path + "trained/model_features_decoder.pt", map_location = self.device))

        self.model_encoder.eval()
        self.model_decoder.eval()
    
