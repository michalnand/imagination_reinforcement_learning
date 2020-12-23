import torch
import torch.nn as nn

from torchviz import make_dot

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, rollouts, hidden_count = 256):
        super(Model, self).__init__()

        self.device = "cpu"

        self.layers_encoder = [ 
            nn.Conv1d(input_shape[0], 32, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32*rollouts, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, input_shape[0])
        ]

        torch.nn.init.xavier_uniform_(self.layers_encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.layers_encoder[3].weight)
        torch.nn.init.xavier_uniform_(self.layers_encoder[5].weight)
       
        self.model_encoder = nn.Sequential(*self.layers_encoder)
        self.model_encoder.to(self.device)

        self.layers_critic = [ 
            nn.Linear(input_shape[0]*2 + outputs_count, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, hidden_count//2),
            nn.ReLU(),            
            nn.Linear(hidden_count//2, 1)           
        ]  

        torch.nn.init.xavier_uniform_(self.layers_critic[0].weight)
        torch.nn.init.xavier_uniform_(self.layers_critic[2].weight)
        torch.nn.init.uniform_(self.layers_critic[4].weight, -0.003, 0.003)
 
        self.model_critic = nn.Sequential(*self.layers_critic) 
        self.model_critic.to(self.device)


       
        print("model_critic")
        print(self.model_encoder)
        print(self.model_critic)
        print("\n\n")
       

    def forward(self, state, states_imagined, action):
        encoder_output = self.model_encoder(states_imagined.transpose(1, 2))

        x = torch.cat([state, encoder_output, action], dim = 1)
        return self.model_critic(x)

     
    def save(self, path):
        torch.save(self.model_critic.state_dict(), path + "trained/model_critic.pt")
        torch.save(self.model_encoder.state_dict(), path + "trained/model_encoder.pt")

    def load(self, path):       
        self.model_critic.load_state_dict(torch.load(path + "trained/model_critic.pt", map_location = self.device))
        self.model_critic.eval()  

        self.model_encoder.load_state_dict(torch.load(path + "trained/model_encoder.pt", map_location = self.device))
        self.model_encoder.eval()  
    
if __name__ == "__main__":
    state_shape     = (26, )
    rollouts        = 16
    outputs_count   = 10
    batch_size      = 64

    model = Model(state_shape, outputs_count, rollouts)

    states_imagined = torch.randn((batch_size, rollouts) + state_shape)
    action          = torch.randn((batch_size, outputs_count))
    state           = torch.randn((batch_size, ) + state_shape)

    y = model(state, states_imagined, action)

    print(y.shape)

    make_dot(y).render("model_critic", format="png")

'''
import model_forward
import model_actor

if __name__ == "__main__":
    state_shape     = (26, )
    rollouts        = 16
    outputs_count   = 10
    batch_size      = 64

    model_forward   = model_forward.Model(state_shape, outputs_count)
    model_actor     = model_actor.Model(state_shape, outputs_count, rollouts)
    model_critic    = Model(state_shape, outputs_count, rollouts)
    state           = torch.randn((batch_size, ) + state_shape)
    action          = torch.randn((batch_size, outputs_count))


    states_imagined = torch.randn((rollouts, batch_size) + state_shape)

    for r in range(rollouts):
        action = model_actor(state)
        states_imagined[r] = model_forward(state, action)

    states_imagined = states_imagined.transpose(0, 1)

    y = model_critic(state, states_imagined, action)

    print(y.shape)

    make_dot(y).render("model_critic", format="png")
'''