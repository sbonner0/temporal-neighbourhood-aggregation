import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GraphConvolution, VariationalGraphConvolution, TemporalGraphConvolution, Base_GCN_RNN

class TNA_model(Base_GCN_RNN):
    """Temporal Neighbourhood Aggregation"""

    def __init__(self, input_dim, n_hidden, n_latent, dropout, bias, device, rnn_model="GRU", use_ln=1, use_skip=1):
        super().__init__()

        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.bias = bias
        self.rnn_model = rnn_model
        self.device = device
        self.use_ln = bool(use_ln)
        self.use_skip = bool(use_skip)

        # Parameters
        self.pos_weight = 0.
        self.norm = 0.

        # GCN Section
        self.gc1 = TemporalGraphConvolution(self.input_dim, self.n_hidden, self.input_dim, self.rnn_model, self.bias, self.use_ln, self.use_skip)
        self.gc2 = TemporalGraphConvolution(self.n_hidden, self.n_latent, self.input_dim, self.rnn_model, self.bias, self.use_ln, self.use_skip)
        self.gc_var = VariationalGraphConvolution(self.n_latent, self.n_latent, self.input_dim, self.device, self.bias)
        self.dropout = dropout

    def reset_hidden_states(self, device):
        """"Reset the hidden state of the RNN"""

        self.gc1.reset_hidden_states(device)
        self.gc2.reset_hidden_states(device)

    def encode_graph(self, x, adj):
        # Perform the encoding stage using a two layer GCN

        x = self.gc1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        z = self.gc_var(x, adj)

        return z