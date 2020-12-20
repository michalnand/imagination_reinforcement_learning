import torch
import torch.nn as nn


class Head(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count, device):
        super(Head, self).__init__()


        self.device = device
        
        self.layers = [
            nn.Linear(input_shape[0] + outputs_count, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, hidden_count),
            nn.ReLU(),           
            nn.Linear(hidden_count, input_shape[0])
        ]

        torch.nn.init.xavier_uniform_(self.layers[0].weight)
        torch.nn.init.xavier_uniform_(self.layers[2].weight)
        torch.nn.init.xavier_uniform_(self.layers[4].weight)
       
        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print(self.model)

    def forward(self, x):
        return self.model(x) + state

    def save(self, file_name_prefix):
        torch.save(self.model.state_dict(), file_name_prefix + ".pt")
        
    def load(self, file_name_prefix):       
        self.model.load_state_dict(torch.load(file_name_prefix + ".pt", map_location = self.device))
        self.model.eval()  
        

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 256, heads_count = 4):
        super(Model, self).__init__()

        self.input_shape    = input_shape
        self.heads_count    = heads_count

        self.device = "cpu"

        self.layers_attention = [
            nn.Linear(input_shape[0] + outputs_count, hidden_count),
            nn.ReLU(),
            nn.Linear(hidden_count, hidden_count),
            nn.ReLU(),           
            nn.Linear(hidden_count, heads_count),
            nn.Softmax(dim=1)
        ]

        torch.nn.init.xavier_uniform_(self.layers_attention[0].weight)
        torch.nn.init.xavier_uniform_(self.layers_attention[2].weight)
        torch.nn.init.xavier_uniform_(self.layers_attention[4].weight)
       
        self.model_attention = nn.Sequential(*self.layers_attention)
        self.model_attention.to(self.device)

        print("model_forward")
        print(self.model_attention)

        self.heads = []
        for h in range(self.heads_count):
            head = Head(input_shape, outputs_count, hidden_count, self.device)
            head.to(self.device)
            self.heads.append(head)

        print("\n\n")
       
    def forward(self, state, action):
        batch_size = state.shape[0]

        x = torch.cat([state, action], dim = 1) 
        
        s       = self.model_attention(x)
        heads   = torch.zeros((self.heads_count, batch_size, ) + self.input_shape)

        for h in range(len(self.heads)):
            heads[h] = self.heads[h](x)

        heads   = heads.transpose(0, 1)
        s       = s.unsqueeze(2).repeat(1, 1, state.shape[1])
        y       = (s*heads).sum(dim=1)
        
        return y

    def save(self, path):
        torch.save(self.model_attention.state_dict(), path + "trained/model_forward_attention.pt")
        for h in range(len(self.heads)):
            self.heads[h].save(path + "trained/model_forward_head_" + str(h))
        
    def load(self, path):       
        self.model_attention.load_state_dict(torch.load(path + "trained/model_forward_attention.pt", map_location = self.device))
        self.model_attention.eval()  

        for h in range(len(self.heads)):
            self.heads[h].load(path + "trained/model_forward_head_" + str(h))
        

if __name__ == "__main__":
    input_shape     = (26, )
    outputs_count   = 8
    batch_size      = 64

    model   = Model(input_shape, outputs_count)

    state   = torch.randn((batch_size, ) + input_shape)
    action  = torch.randn((batch_size, outputs_count))

    y = model(state, action)

    model.save("./")
    model.load("./")
    print(y.shape)
