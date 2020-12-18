import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import numpy
from torchviz import make_dot


class GConv(MessagePassing):
    def __init__(self, in_channels, out_channels, device = "cpu"):
        super(GConv, self).__init__(aggr='add')

        self.device = device
        self.in_channels  = in_channels
        self.out_channels = out_channels

        self.linear_layer = torch.nn.Linear(in_channels, out_channels)
        torch.nn.init.xavier_uniform_(self.linear_layer.weight)
        
        self.linear_layer.to(self.device)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(1))

        # Step 2: Multiply with weights
        x = self.linear_layer(x) 

        # Step 3: Calculate the normalization
        row, col        = edge_index
        deg             = degree(row, x.size(1), dtype=x.dtype)
        deg_inv_sqrt    = deg.pow(-0.5)
        norm            = deg_inv_sqrt[row]*deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(1), x.size(1)), x=x, norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j

    def save(self, file_name):
        torch.save(self.linear_layer.state_dict(), file_name)

    def load(self, file_name):
        self.linear_layer.load_state_dict(torch.load(file_name, map_location = "cpu"))
        self.linear_layer.eval()  

    def __repr__(self):
        return "GraphConv(in_channels={}, out_channels={})".format(self.in_channels, self.out_channels)
    

class GConvSeq(torch.nn.Module):
    def __init__(self, channels_list, device = "cpu"):
        super(GConvSeq, self).__init__()

        self.layers = []
        for i in range(len(channels_list)-1):
            gconv     = GConv(channels_list[i], channels_list[i+1], device)
            act       = torch.nn.ReLU()
            act       = act.to(device)

            self.layers.append(gconv)
            self.layers.append(act)

    def forward(self, x, edge_index):
        for i in range(len(self.layers)):
            if i%2 == 0:
                x = self.layers[i](x, edge_index)
            else:
                x = self.layers[i](x)

        return x

    def save(self, file_name_prefix):
        for i in range(len(self.layers)):
            if i%2 == 0:
                self.layers[i].save(file_name_prefix + str(i) + ".pt")

    def load(self, file_name_prefix):
        for i in range(len(self.layers)):
            if i%2 == 0:
                self.layers[i].load(file_name_prefix + str(i) + ".pt")

    def __repr__(self):

        s = ""
        for layer in self.layers:
            s+= str(layer) + "\n"

        return s

if __name__ == "__main__":
    batch_size      = 5
    inputs_count    = 25
    nodes_count     = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #random edges, 1/0 connections
    edges           = torch.from_numpy(numpy.random.randint(2, size=(2, nodes_count))).to(device)

    #random input
    x               = torch.randn((batch_size, nodes_count, inputs_count)).to(device)

    #create model, 2 hidden layers
    model = GConvSeq(channels_list=[inputs_count, 256, 256], device=device)
    print(model)

    #forward truth model
    y = model(x, edges)

    print("y_shape = ", y.shape)

    make_dot(y).render("model", format="png")