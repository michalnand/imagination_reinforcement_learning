import torch
import torch.nn as nn

from torchviz import make_dot

class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 256, layers_count=1):
        super(Model, self).__init__()

        #self.device = "cpu"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers_count   = layers_count
        self.hidden_count   = hidden_count
        
        self.model_lstm     = nn.LSTM(input_shape[0] + outputs_count, self.hidden_count, num_layers=self.layers_count, batch_first=True)
        self.model_output   = nn.Linear(self.hidden_count, input_shape[0])

        torch.nn.init.xavier_uniform_(self.model_output.weight)

        self.model_lstm.to(self.device)
        self.model_output.to(self.device)
       
        print("model_forward")
        print(self.model_lstm)
        print(self.model_output)
        print("\n\n")
       
    def forward(self, state_seq, action_seq):
        #cat state and action vectors
        x = torch.cat([state_seq, action_seq], dim = 2) 

        #initial cell state
        h0 = torch.zeros(self.layers_count, x.size(0), self.hidden_count).requires_grad_().to(x.device)
        c0 = torch.zeros(self.layers_count, x.size(0), self.hidden_count).requires_grad_().to(x.device)

        l_out, (hidden, cell) = self.model_lstm(x, (h0.detach(), c0.detach()))

        #l_out shape = (batch, sequence, hidden_units)
        #take last hidden state
        l_out = l_out[:, -1, :]

        #last hidden state is input into output layer
        y = self.model_output(l_out)

        #add skip connection 
        y = y + state_seq[:, -1, :]

        return y

    def save(self, path):
        torch.save(self.model_lstm.state_dict(), path + "trained/model_forward_lstm.pt")
        torch.save(self.model_output.state_dict(), path + "trained/model_forward_output.pt")

    def load(self, path):       
        self.model_lstm.load_state_dict(torch.load(path + "trained/model_forward_lstm.pt", map_location = self.device))
        self.model_lstm.eval()  

        self.model_output.load_state_dict(torch.load(path + "trained/model_forward_output.pt", map_location = self.device))
        self.model_output.eval()  
    

if __name__ == "__main__":
    state_shape     = (20, )
    actions_count   = 7
    batch_size      = 64
    sequence_length = 8

    model = Model(state_shape, actions_count)

    state_seq   = torch.randn((batch_size, sequence_length, ) + state_shape)
    action_seq  = torch.randn((batch_size, sequence_length, actions_count))

    y = model(state_seq, action_seq)

    make_dot(y).render("model", format="png")