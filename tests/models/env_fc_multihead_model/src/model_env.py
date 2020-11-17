import torch
import torch.nn as nn

class CuriosityHead(torch.nn.Module):

    def __init__(self, input_shape, outputs_count, hidden_count = 128):
        super(CuriosityHead, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        self.model_features = nn.Sequential(
                                            nn.Linear(input_shape[0] + outputs_count, hidden_count),
                                            nn.ReLU()
        )


        self.model_state = nn.Sequential(
                                            nn.Linear(hidden_count, hidden_count//2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_count//2, input_shape[0])
        )

        self.model_reward = nn.Sequential(
                                            nn.Linear(hidden_count, hidden_count//2),
                                            nn.ReLU(),
                                            nn.Linear(hidden_count//2, 1)
        ) 


        self.model_features.to(self.device)
        self.model_state.to(self.device)
        self.model_reward.to(self.device)

        print(self.model_features)
        print(self.model_state)
        print(self.model_reward)
        print("\n\n\n")

    def forward(self, input):
        features = self.model_features(input) 

        state_prediction = self.model_state(features)

        return state_prediction, self.model_reward(features)

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

    def __init__(self, input_shape, outputs_count, hidden_count = 128, n_heads = 4):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_heads = n_heads

        heads = []
        for i in range(n_heads):
            heads.append(CuriosityHead(input_shape, outputs_count, hidden_count))
        
        self.curiosity_heads = ListModules(*heads)

        self.input_shape = input_shape

        self.model_attention = nn.Sequential(
                                                nn.Linear(input_shape[0] + outputs_count, hidden_count),
                                                nn.ReLU(),
                                                nn.Linear(hidden_count, hidden_count//2),
                                                nn.ReLU(),
                                                nn.Linear(hidden_count//2, self.n_heads),
                                                nn.Softmax(dim=1)
        )


        self.model_attention.to(self.device)
        print(self.model_attention)
        print("\n\n\n")

    def forward(self, state, action):
        batch_size = state.shape[0]
        x = torch.cat([state, action], dim = 1).detach()

        attention = self.model_attention.forward(x)
       
        heads_output_state  = torch.zeros((self.n_heads, batch_size) + self.input_shape)
        heads_output_reward = torch.zeros((self.n_heads, batch_size, 1))

        for i in range(self.n_heads):            
            heads_output_state[i], heads_output_reward[i] = self.curiosity_heads[i].forward(x)

      
        heads_output_state  = heads_output_state.transpose(0, 1)
        heads_output_reward = heads_output_reward.transpose(0, 1)

        attention_state = attention.unsqueeze(-1).repeat((1, 1, ) + self.input_shape)
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
        