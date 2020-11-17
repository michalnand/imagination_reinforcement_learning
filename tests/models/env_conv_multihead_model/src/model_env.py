import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CuriosityHead(torch.nn.Module):
    def __init__(self, input_shape, outputs_count):
        super(CuriosityHead, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_shape    = input_shape 
        self.outputs_count  = outputs_count
        
        input_channels  = self.input_shape[0]
        input_height    = self.input_shape[1]
        input_width     = self.input_shape[2]   

        kernels_count = 32

        fc_input_height = input_shape[1]//(4*2*2)
        fc_input_width  = input_shape[2]//(4*2*2)

        self.conv0 = nn.Sequential( 
                                    nn.Conv2d(input_channels + outputs_count, kernels_count, kernel_size=4, stride=4, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(kernels_count, kernels_count, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
                                    nn.Conv2d(kernels_count, kernels_count, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
        )

        self.deconv0 = nn.Sequential(
                                        nn.ConvTranspose2d(kernels_count, input_channels, kernel_size=4, stride=4, padding=0),
                                    )

        self.reward = nn.Sequential(
                                            nn.Conv2d(kernels_count, kernels_count, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(),

                                            nn.Conv2d(kernels_count, kernels_count, kernel_size=3, stride=2, padding=1),
                                            nn.ReLU(),

                                            Flatten(),
                                            nn.Linear(fc_input_height*fc_input_width*kernels_count, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, 1)
        ) 

        self.conv0.to(self.device)
        self.conv1.to(self.device) 
        self.deconv0.to(self.device) 
        self.reward.to(self.device) 

        print(self.conv0)
        print(self.conv1)
        print(self.deconv0)  
        print(self.reward)                      

    def forward(self, model_input):
        conv0_output     = self.conv0(model_input)
        conv1_output     = self.conv1(conv0_output)

        tmp = conv0_output + conv1_output

        observation_prediction = self.deconv0(tmp)
        reward_prediction      = self.reward(tmp)
        
        return observation_prediction, reward_prediction
    
    def _layers_to_model(self, layers):

        for i in range(len(layers)):
            if isinstance(layers[i], nn.Conv2d) or isinstance(layers[i], nn.Linear):
                torch.nn.init.xavier_uniform_(layers[i].weight)

        model = nn.Sequential(*layers)
        model.to(self.device)

        return model


class ListModules(nn.Module):
    def __init__(self, *args):
        super(ListModules, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class Model(torch.nn.Module):

    def __init__(self, input_shape, actions_count, n_heads = 4):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_heads = n_heads

        heads = []
        for i in range(n_heads):
            heads.append(CuriosityHead(input_shape, actions_count))
        
        self.curiosity_heads = ListModules(*heads)

        self.input_shape = input_shape

        input_channels     = input_shape[0]
        input_height       = input_shape[1]
        input_width        = input_shape[2]

        self.model_attention = nn.Sequential(
                                                nn.Conv2d(input_channels + actions_count, 32, kernel_size=3, stride=1, padding=1),
                                                nn.ReLU(),
                                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                                                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                                nn.ReLU(),
                                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                                                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                                nn.ReLU(),
                                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                                                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                                                nn.ReLU(),
                                                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

                                                Flatten(),

                                                nn.Linear(32*(input_height//16)*(input_width//16), 128),
                                                nn.ReLU(),

                                                nn.Linear(128, self.n_heads),
                                                nn.Softmax(dim=1)
        )


        self.model_attention.to(self.device)

        print(self.model_attention)
        print("\n\n\n")

    def forward(self, state, action):
        batch_size = state.shape[0]

        action_ = action.unsqueeze(1).unsqueeze(1).transpose(3, 1).repeat((1, 1, self.input_shape[1], self.input_shape[2])).to(self.device)
        x       = torch.cat([state, action_], dim = 1)
    
        attention = self.model_attention.forward(x)
       
        heads_output_state  = torch.zeros((self.n_heads, batch_size) + self.input_shape).to(self.device)
        heads_output_reward = torch.zeros((self.n_heads, batch_size, 1)).to(self.device)

        for i in range(self.n_heads):            
            heads_output_state[i], heads_output_reward[i] = self.curiosity_heads[i].forward(x)

      
        heads_output_state  = heads_output_state.transpose(0, 1)
        heads_output_reward = heads_output_reward.transpose(0, 1)

        attention_state = attention.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat((1, 1, ) + self.input_shape)
        attention_reward= attention.unsqueeze(-1)

        state_predicted  = torch.sum(attention_state*heads_output_state, dim = 1)
        reward_predicted = torch.sum(attention_reward*heads_output_reward, dim = 1)


        return state_predicted + state.detach(), reward_predicted

    def save(self, path):
        torch.save(self.model_attention.state_dict(), path + "trained/model_curiosity_attention.pt")
        for i in range(self.n_heads):
            torch.save(self.curiosity_heads[i].state_dict(), path + "trained/model_curiosity_head_" + str(i) + ".pt")

        

    def load(self, path):       
        self.model_attention.load_state_dict(torch.load(path + "trained/model_curiosity_attention.pt", map_location = self.device))
        self.model_attention.eval() 

        for i in range(self.n_heads):
            self.curiosity_heads[i].load_state_dict(torch.load(path + "trained/model_curiosity_head_" + str(i) + ".pt", map_location = self.device))
            self.curiosity_heads[i].eval()
        

from torchviz import make_dot


if __name__ == "__main__":
    batch_size      = 1
    input_shape     = (4, 96, 96)
    actions_count   = 5

    model = Model(input_shape, actions_count, n_heads = 32)

    state       = torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2])
    action      = torch.randn(batch_size, actions_count)

    state_predicted, reward_predicted = model.forward(state, action)
    
    make_dot(state_predicted).render("graph", format="png")