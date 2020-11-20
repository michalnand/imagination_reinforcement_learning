import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Model(torch.nn.Module):

    def __init__(self, input_shape, outputs_count):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape

        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]    


        fc_inputs_count = 128*input_width*input_height

               
        self.layers = [ 
            nn.Conv2d(self.input_shape[0] + outputs_count, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

            Flatten(),

            nn.Linear(in_features=fc_inputs_count, out_features=128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=1)
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
        action_ = action.unsqueeze(1).unsqueeze(1).transpose(3, 1).repeat((1, 1, self.input_shape[1], self.input_shape[2])).to(self.device)

        x = torch.cat([state, action_], dim=1)

        return self.model(x)

    def save(self, path):
        print("saving ", path)

        torch.save(self.model.state_dict(), path + "trained/model_reward.pt")

    def load(self, path):
        print("loading ", path) 

        self.model.load_state_dict(torch.load(path + "trained/model_reward.pt", map_location = self.device))
        self.model.eval() 

if __name__ == "__main__":
    batch_size = 8

    channels = 128
    height   = 6
    width    = 6

    actions_count = 9


    state   = torch.rand((batch_size, channels, height, width))
    action  = torch.rand((batch_size, actions_count))

    model = Model((channels, height, width), actions_count)

    y = model.forward(state, action)

    print(y.shape)


