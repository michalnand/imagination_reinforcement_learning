import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 256):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.robots_count       = input_shape[0]
        self.features_count     = input_shape[1]
        self.actions_count      = outputs_count//self.robots_count


        self.layers = [ 
            nn.Linear(self.features_count + self.actions_count, hidden_count),
            nn.ReLU(),            
            
            nn.Linear(hidden_count, hidden_count//2),
            nn.ReLU(),   

            nn.Linear(hidden_count//2, self.features_count)
        ]
       
        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)
        torch.nn.init.uniform_(self.layers[4].weight, -0.003, 0.003)
 
        self.model = nn.Sequential(*self.layers) 
        self.model.to(self.device)

        print("model_forward")
        print(self.model)
        print("\n\n")
       

    def forward(self, state, action):
        batch_size      = state.shape[0]
        
        state_  = state.reshape((batch_size*self.robots_count , self.features_count))
        action_ = action.reshape((batch_size*self.robots_count, self.actions_count))

        x   = torch.cat([state_, action_], dim = 1) 
        
        y_      = self.model(x)
        
        y       = y_.reshape((batch_size, self.robots_count, self.features_count))

        return y + state


    def save(self, path):
        print("saving to ", path)
        torch.save(self.model.state_dict(), path + "trained/model_forward.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model.load_state_dict(torch.load(path + "trained/model_forward.pt", map_location = self.device))
        self.model.eval()  



if __name__ == "__main__":
    batch_size      = 64
    robots_count    = 10
    features_count  = 26

    input_shape     = (robots_count, features_count)
    outputs_count   = 5*robots_count

    model = Model(input_shape, outputs_count)

    state   = torch.randn((batch_size, ) + input_shape)
    action  = torch.randn((batch_size, outputs_count))

    y = model.forward(state, action)

    print(y.shape)
