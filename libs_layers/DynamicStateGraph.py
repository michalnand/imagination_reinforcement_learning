import numpy
import matplotlib.pyplot as plt
import networkx as nx

import torch

class DynamicGraphState:

    def __init__(self, state_size, alpha = 0.01):
        self.state_size  = state_size
        self.correlation = numpy.zeros((state_size, state_size))
        self.alpha       = alpha


    def train(self, state):
        state_a = numpy.expand_dims(state, 0)
        state_b = numpy.transpose(state_a)

        correlation = state_a*state_b
        correlation = correlation**2 + (10**-6)

        self.correlation = (1.0 - self.alpha)*self.correlation + self.alpha*correlation

        adjacency_matrix = 1.0/self.correlation
        
        self.graph = self._minimum_spanning_tree(adjacency_matrix)

        m0 = numpy.transpose(numpy.matrix(self.graph.edges))
        m1 = numpy.concatenate([m0[1], m0[0]])

        self.edge_index         = numpy.concatenate([m0, m1], axis=1)
        self.adjacency_matrix   = numpy.array(numpy.sign(nx.adjacency_matrix(self.graph).todense()))

    def train_batch(self, state_batch):
        batch_size      = state_batch.shape[0]
        for b in range(batch_size):
            self.train(state_batch[b])

    def eval(self, state):
        state_          = numpy.expand_dims(state, 0)
        state_masked    = state_*self.adjacency_matrix
        
        return state_masked, self.adjacency_matrix, self.edge_index

    def eval_batch(self, state_batch):
        batch_size      = state_batch.shape[0]
        state_masked    = numpy.zeros((batch_size,  self.state_size , self.state_size))

        for b in range(batch_size):
            state_masked[b], _, _ = self.eval(state_batch[b])

        return state_masked, self.adjacency_matrix, self.edge_index

    def eval_torch(self, state, train = False):

        device = state.device

        state_np = state.detach().to("cpu").numpy()
        if train:
            self.train_batch(state_np)

        state_masked, adjacency_matrix, edge_index = self.eval_batch(state_np)

        state_masked = torch.from_numpy(state_masked).to(device).float()
        edge_index   = torch.from_numpy(edge_index).to(device)

        return state_masked, edge_index


    def _minimum_spanning_tree(self, adjacency_matrix):
        gr      = nx.from_numpy_matrix(numpy.matrix(adjacency_matrix))
        result  = nx.minimum_spanning_tree(gr, weight="weight")
        return result

    def show_graph(self):
        plt.clf()

        labels = {}
        for i in range(self.correlation.shape[0]):
            labels[i] = str(i)

        nx.draw(self.graph, node_size=500, labels=labels, with_labels=True)
        plt.ion()
        plt.show()
        plt.pause(0.001)
        
    
if __name__ == "__main__":
    state_size  = 32
    dg          = DynamicGraphState(state_size)

    i = 0
    while True:
        state = numpy.random.randn(state_size)
        
        dg.train(state)

        state_batch = numpy.random.randn(5, state_size)

        #state_masked, adjacency_matrix, edge_index = dg.eval(state)
        state_masked, adjacency_matrix, edge_index = dg.eval_batch(state_batch)

        if i%100 == 0:
            #print(numpy.round(adjacency_matrix, 3),"\n")
            print(numpy.round(state_masked, 3),"\n\n\n\n")

            dg.show_graph()

        i+=1
            

