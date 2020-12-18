import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../../../../..')
#sys.path.insert(0, '../../../../../..')

import libs_layers

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 256):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        size = input_shape[0] + outputs_count
        
        self.graph = libs_layers.DynamicGraphState(size, alpha=0.001)

        self.model_graph    = libs_layers.GConvSeq([size, hidden_count, hidden_count], self.device)
        
        self.layers_output  = [
            nn.Linear(size*hidden_count, 1)
        ]
        
        torch.nn.init.uniform_(self.layers_output[0].weight, -0.003, 0.003)

        self.model_output = nn.Sequential(*self.layers_output)
        self.model_output.to(self.device)

        print("model_critic")
        print(self.model_graph)
        print(self.model_output)
        print("\n\n")
       

    def forward(self, state, action):
        x = torch.cat([state, action], dim = 1)

        state_masked, edge_index = self.graph.eval_torch(x, train=True)
        
        y = self.model_graph(state_masked, edge_index)
 
        y = y.reshape((y.shape[0], y.shape[1]*y.shape[2]))

        y = self.model_output(y)
        return y

     
    def save(self, path):
        print("saving to ", path)
        
        self.model_graph.save("trained/model_critic_graph_")
        torch.save(self.model_output.state_dict(), path + "trained/model_critic_output.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model_graph.load("trained/model_critic_graph_")
        self.model_graph.eval()

        self.model_output.load_state_dict(torch.load(path + "trained/model_critic_output.pt", map_location = self.device))
        self.model_output.eval()  
    
if __name__ == "__main__":
    input_shape     = (25, )
    outputs_count   = 8
    batch_size      = 32

    model = Model(input_shape, outputs_count)

    x = torch.randn((batch_size, ) + input_shape).to(model.device)
    a = torch.randn((batch_size, outputs_count)).to(model.device)

    y = model(x, a)

    print(">>> ", y.shape)

