import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bool(bias):
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        torch.nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class VariationalGraphConvolution(Module):
    """
    Variational GCN layer
    """

    def __init__(self, in_features, out_features, N, device,  bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.N = N
        self.bias = bias
        self.device = device
        self.gcn_log_sig = GraphConvolution(self.in_features, self.out_features, self.bias)
        self.gcn_mu = GraphConvolution(self.in_features, self.out_features, self.bias)

    def forward(self, input, adj):

        self.mu = self.gcn_mu(input, adj)
        self.log_sig = self.gcn_log_sig(input, adj)
        z = self.mu + (torch.randn(self.N, self.out_features, device=self.device) * torch.exp(self.log_sig))

        return z

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class TemporalGraphConvolution(Module):
    """
    GCN Layer combinned with an RNN and optional LayerNorm.
    """

    def __init__(self, in_features, out_features, hidden_size_in, rnn_model="RNN", bias=True, layer_norm=True, skip_connection=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size_in = hidden_size_in
        self.bias = bias
        self.ln_active = layer_norm
        self.skip_connection = skip_connection

        # GCN Section
        self.gcn = GraphConvolution(self.in_features, self.out_features, self.bias)
        self.gcn_layer_norm = nn.LayerNorm(self.out_features)
        self.skip = nn.Linear(self.out_features*2, self.out_features)

        # RNN Section
        self.rnn_model = rnn_model
        if self.rnn_model == "GRU":
            self.rnn = nn.GRUCell(self.out_features, self.out_features)
            self.previous_h_t = self.init_hidden()
            if self.ln_active:
                self.layer_norm = nn.LayerNorm(self.out_features)

        elif self.rnn_model == "LSTM":
            self.rnn = nn.LSTMCell(self.out_features, self.out_features)
            self.previous_h_t, self.previous_c_t = self.init_hidden()
            if self.ln_active:
                self.layer_norm_h = nn.LayerNorm(self.out_features)
                self.layer_norm_c = nn.LayerNorm(self.out_features)

        elif self.rnn_model == "RNN":
            self.rnn = nn.RNNCell(self.out_features, self.out_features)
            self.previous_h_t = self.init_hidden()
            if self.ln_active:
                self.layer_norm = nn.LayerNorm(self.out_features)

    def init_hidden(self):
        """Initialise the hidden state of the RNN"""

        if self.rnn_model == "LSTM":
            hidden = torch.zeros(self.hidden_size_in, self.out_features, requires_grad=False)
            cell = torch.zeros(self.hidden_size_in, self.out_features, requires_grad=False)

            return hidden, cell

        else:
            hidden = torch.zeros(self.hidden_size_in, self.out_features, requires_grad=False)

            return hidden
    
    def reset_hidden_states(self, device):
        """"Reset the hidden state of the RNN"""

        if self.rnn_model == "LSTM":
            
            self.previous_h_t, self.previous_c_t = self.init_hidden()
            self.previous_h_t = self.previous_h_t.to(device)
            self.previous_c_t = self.previous_c_t.to(device)

        else:
            
            self.previous_h_t = self.init_hidden()
            self.previous_h_t = self.previous_h_t.to(device)

    def forward(self, input, adj):

        x = F.relu(self.gcn(input, adj))
        if self.ln_active:
            x = self.gcn_layer_norm(x)

        # Pass the hidden representation through an RNN 
        if self.rnn_model == "LSTM":
            self.previous_h_t, self.previous_c_t = self.rnn(x, (self.previous_h_t, self.previous_c_t))
            if self.ln_active:
                self.previous_h_t = self.layer_norm_h(self.previous_h_t)
                self.previous_c_t = self.layer_norm_c(self.previous_c_t)
        else:
            self.previous_h_t = self.rnn(x, self.previous_h_t)
            if self.ln_active:
                self.previous_h_t = self.layer_norm(self.previous_h_t)
        
        # If enabled, combine the GCN and RNN representations
        if self.skip_connection:
            x = torch.cat((self.previous_h_t, x), 1)
            x = F.leaky_relu(self.skip(x))
        else:
            x = self.previous_h_t

        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Base_GCN_RNN(nn.Module):
    """Base Class for Generic Graph Auto Encoder"""

    def __init__(self):
        super().__init__()

    def reset_hidden_states(self, device):
        """"Reset the hidden state of the RNN"""

        pass

    def encode_graph(self, x, adj):
        # Perform the encoding stage using a two layer GCN

        pass

    def decode_graph(self, z):
        # Here the reconstruction is based upon the inner product between the latent representation
        adj_hat = torch.mm(z, z.t())

        return adj_hat

    def forward(self, x, adj):
        """"Input Adjacency matrix and feature matrix for the graph and return the predicted graph at T+1"""

        # Encoder Step
        z = self.encode_graph(x, adj) # x = |V| x n_latent matrix

        # Decoder Step
        adj_hat = self.decode_graph(z) # adj_hat = |V| x |V|

        return adj_hat

    def get_embeddings(self, x, adj):

        return self.encode_graph(x, adj)