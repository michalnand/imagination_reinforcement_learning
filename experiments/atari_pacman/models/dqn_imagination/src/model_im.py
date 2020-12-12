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
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]

        self.features_count = 32*(input_height//16)*(input_width//16)

        self.layers_features = [
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten() 
        ]

        self.layers_inverse = [
            nn.Linear(2*self.features_count, 512),
            nn.ReLU(),
            nn.Linear(512, outputs_count)
        ]

        self.layers_forward = [
            nn.Linear(self.features_count + outputs_count, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.features_count)
        ]

        for i in range(len(self.layers_features)):
            if hasattr(self.layers_features[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_features[i].weight)

        for i in range(len(self.layers_inverse)):
            if hasattr(self.layers_inverse[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_inverse[i].weight)

        for i in range(len(self.layers_forward)):
            if hasattr(self.layers_forward[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers_forward[i].weight)

        self.features_model = nn.Sequential(*self.layers_features)
        self.features_model.to(self.device)

        self.inverse_model = nn.Sequential(*self.layers_inverse)
        self.inverse_model.to(self.device)

        self.forward_model = nn.Sequential(*self.layers_forward)
        self.forward_model.to(self.device)

        print("model_forward")
        print(self.features_model)
        print(self.inverse_model)
        print(self.forward_model)
        print("\n\n")

    def forward(self, state, state_next, action):
        features_state      = self.features_model(state)
        features_state_next = self.features_model(state_next)

        action_predicted    = self.inverse_model(torch.cat([features_state, features_state_next], dim=1))

        features_state_next_predicted = self.forward_model(torch.cat([features_state, action], dim=1))

        return action_predicted, features_state_next, features_state_next_predicted

    def eval_next_features(self, state, action):
        features_state      = self.features_model(state)
        features_state_next_predicted = self.forward_model(torch.cat([features_state, action], dim=1))

        return features_state, features_state_next_predicted


    def save(self, path):
        torch.save(self.features_model.state_dict(), path + "trained/features_model.pt")
        torch.save(self.inverse_model.state_dict(), path + "trained/inverse_model.pt")
        torch.save(self.forward_model.state_dict(), path + "trained/forward_model.pt")

    def load(self, path):
        self.features_model.load_state_dict(torch.load(path + "trained/features_model.pt", map_location = self.device))
        self.inverse_model.load_state_dict(torch.load(path + "trained/inverse_model.pt", map_location = self.device))
        self.forward_model.load_state_dict(torch.load(path + "trained/forward_model.pt", map_location = self.device))

        self.features_model.eval() 
        self.inverse_model.eval() 
        self.forward_model.eval() 

if __name__ == "__main__":
    batch_size = 8

    channels = 3
    height   = 96
    width    = 96

    actions_count = 9


    state           = torch.rand((batch_size, channels, height, width))
    state_next      = torch.rand((batch_size, channels, height, width))
    action          = torch.rand((batch_size, actions_count))

    model = Model((channels, height, width), actions_count)

    action_predicted, features_state_next, features_state_next_predicted = model.forward(state, state_next, action)

    print(action_predicted.shape, features_state_next.shape, features_state_next_predicted.shape)


