import torch
import torch.nn as nn

import sys
sys.path.insert(0, '../../../../..')

import libs_layers


class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 256):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inputs_count = input_shape[0] + outputs_count
        
        self.graph = libs_layers.DynamicGraphState(inputs_count, alpha=0.001)

        self.model_graph    = libs_layers.GConvSeq([inputs_count, hidden_count, hidden_count], self.device)
        
        self.layers_output  = [
            nn.Linear(hidden_count, input_shape[0])
        ]
        
        torch.nn.init.xavier_uniform_(self.layers_output[0].weight)

        self.model_output = nn.Sequential(*self.layers_output)
        self.model_output.to(self.device)

        print("model_forward")
        print(self.model_graph)
        print(self.model_output)
        print("\n\n")

       
    def forward(self, state, action):
        x = torch.cat([state, action], dim = 1) 

        input_masked, edge_index = self.graph.eval_torch(x, train=True)
        
        y = self.model_graph(input_masked, edge_index)
 
        y = y.reshape((y.shape[0], y.shape[1]*y.shape[2]))

        y = self.model_output(y)

        return y + state


    def save(self, path):
        print("saving to ", path)
        
        self.model_graph.save("trained/model_forward_graph_")
        torch.save(self.model_output.state_dict(), path + "trained/model_forward_output.pt")

    def load(self, path):       
        print("loading from ", path)
        self.model_graph.load("trained/model_forward_graph_")
        self.model_graph.eval()

        self.model_output.load_state_dict(torch.load(path + "trained/model_forward_output.pt", map_location = self.device))
        self.model_output.eval()  
    
