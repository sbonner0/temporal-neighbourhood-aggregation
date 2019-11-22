import os
import csv
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score,
                             roc_auc_score, confusion_matrix)

from data_utils import load_adj_graph
from preprocessing import mask_edges, mask_test_edges, preprocess_graph

# ------------------------------------
# Some functions borrowed from:
# https://github.com/tkipf/pygcn and 
# https://github.com/tkipf/gae
# ------------------------------------


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def eval_gae(edges_pos, edges_neg, adj_rec):
    """Evaluate the model via link prediction"""

    # Predict on test set of edges
    adj_rec = torch.sigmoid(adj_rec)
    adj_rec = adj_rec.cpu().numpy()
    preds = []
    
    # Loop over the positive test edges
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])
    
    preds_neg = []

    # Loop over the negative test edges
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    decision_threshold = 0.55

    accuracy = accuracy_score(labels_all, (preds_all > decision_threshold).astype(float))
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    tn, fp, fn, tp = confusion_matrix(labels_all, (preds_all > decision_threshold).astype(float)).ravel()

    return accuracy, roc_score, ap_score, tn, fp, fn, tp


def make_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def save_model(epoch, model, optimizer, filepath="model.cpt"):
    """Save the model to disk"""

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, filepath)

    print("Model Saved")

def evaluate_on_testset(model, test_edges, test_edges_false, data):
    """Evaluate the model on a given test set of graph edges"""

    with torch.no_grad():
        model.eval()
        adj_rec = model(data['features'], data['adj_norm'])
        accuracy, roc_score, ap_score, tn, fp, fn, tp = eval_gae(test_edges, test_edges_false, adj_rec)
    model.train()

    return accuracy, roc_score, ap_score, tn, fp, fn, tp

def run_test_set(gae, data_loader, device, train_seq_length):
    """Run the test set"""

    gae.eval()
    gae.reset_hidden_states(device)
    # Loop is needed to update the hidden states of the RNNs
    for i in range(train_seq_length):
        data = data_loader.data_list[i]
        data['adj_norm'] = data['adj_norm'].to(device)
        data['features'] = data['features'].to(device)
        logits = gae(data['features'], data['adj_norm'])

    data = data_loader.data_list[-1] # Load the last but one graph in the sequence as the input data
    data['adj_norm'] = data['adj_norm'].to(device)
    data['features'] = data['features'].to(device)

    # Check if there are any new edges
    if data_loader.new_edges_list[-1] is not None:
        accuracy, roc_score, ap_score, tn, fp, fn, tp = evaluate_on_testset(gae, data_loader.new_edges_list[-1], data_loader.new_edges_false_list[-1], data)
    else:
        accuracy, roc_score, ap_score, tn, fp, fn, tp = 0,0,0,0,0,0,0 


    print("Running on Testset From New Edges Only")
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test ROC score: {roc_score:.4f}')
    print(f'Test AP score: {ap_score:.4f}')

    ae_accuracy, ae_roc_score, ae_ap_score, ae_tn, ae_fp, ae_fn, ae_tp = evaluate_on_testset(gae, data_loader.test_edges_list[-1], data_loader.test_edges_false_list[-1], data)


    print("Running on Testset From Whole Graph")
    print(f'Test Accuracy: {ae_accuracy:.4f}')
    print(f'Test ROC score: {ae_roc_score:.4f}')
    print(f'Test AP score: {ae_ap_score:.4f}')

def prepare_data_for_model(adj_train, target_adj_train, device):
    """"Prepare the given data ready to be put in the model"""

    # Some preprocessing
    adj_train_norm   = preprocess_graph(adj_train)
    adj_train_norm   = make_sparse(adj_train_norm)
    adj_train_labels = torch.FloatTensor(target_adj_train + sp.eye(target_adj_train.shape[0]).todense())

    # Features are the identity matrix
    features = sp.eye(adj_train.shape[0]).tolil()
    features = make_sparse(features)

    data = {
        'adj_norm'  : adj_train_norm,
        'adj_labels': adj_train_labels,
        'features'  : features,
    }

    data['adj_norm'] = data['adj_norm'].to(device)
    data['adj_labels'] = data['adj_labels'].to(device)
    data['features'] = data['features'].to(device)

    return data

class GraphData():
    """Class in which to load and process the graph data"""

    def __init__(self, args, num_pairs, new_edge_scaler=250., new_edge_scaling_method='none'):
        super().__init__()

        self.args = args
        self.num_pairs = num_pairs # num_pairs is the sequence length - 1. 

        # Preload the training graphs into memory...not very scaleable but helps with CPU load
        # Preload all but the last graph as this is use for val/test
        self.data_loc = os.path.join(args.data_loc, args.dataset)
        self.data_list = []
        self.new_edge_scaler = new_edge_scaler
        # Loop over all pairs of offset graphs.
        # So if num_pairs = 29, the loop will be from 0 to 28.
        # We load graph i as the training graph and i+1 as the target graph
        # So for evaluation or test, we load the last element in data_list, data_list[-1] as it contains the last input graph in the sequence and we ignore the labels
        for i in range(self.num_pairs):
            adj_train, features = load_adj_graph(f'{self.data_loc}_t{i}.npz') # Load the input graph 
            target_adj_train, target_features = load_adj_graph(f'{self.data_loc}_t{i+1}.npz') # Load the next one in the sequence
            print(f'{args.dataset}_t{i} is the input graph and {args.dataset}_t{i+1} the target')

            # Check the two graphs are the same size
            assert adj_train.shape == target_adj_train.shape
            self.data_list.append(prepare_data_for_model(adj_train, target_adj_train, "cpu"))
        assert len(self.data_list) == self.num_pairs #Should be the length of the time series as the index will start from zero
        # data_list contains all pairs of training/target graphs
        # The last element in data_list contains graph seq_length-2. So if seq_len is 30, data_list will contain graph number 28 
        print("Training graphs loaded into memory")

        # make loss weight matrices:
        N = self.data_list[0]['adj_labels'].shape[0] # number of nodes in graph
        if new_edge_scaling_method == 'decay':

            print("Making loss weights... ")
            print("Stacking... ")
            self.loss_weights = np.stack([g['adj_labels'].numpy() for g in self.data_list])
            print("Cummin and summin... ")
            self.loss_weights = self.loss_weights.cumsum(0)
            print("one overing... ")
            self.loss_weights = 1 + self.new_edge_scaler / self.loss_weights
            # self.loss_weights = 1 + 10 * np.exp(-self.loss_weights)
            print("inf busting... ")
            self.loss_weights[np.isinf(self.loss_weights)] = 1
            print("setting diagonal weights to 1... ")
            eye = np.expand_dims(np.eye(N),0)
            self.loss_weights = self.loss_weights * (1-eye) # zeroing diagonals
            self.loss_weights += eye # oneing diagonals
            print("torchiflying... ")
            self.loss_weights = torch.tensor(self.loss_weights).float()
            print("Loss weights computed")

        elif new_edge_scaling_method == 'burst':

            print("Stacking... ")
            self.loss_weights = np.stack([g['adj_labels'].numpy() for g in self.data_list])
            print("shifting... ")
            self.loss_weights = np.concatenate((np.zeros((1,N,N)), self.loss_weights), axis=0)
            print("subtracting... ")
            self.loss_weights = self.loss_weights[1:] + self.new_edge_scaler * (self.loss_weights[1:] - self.loss_weights[:-1]) + 1 - self.loss_weights[1:]
            print("setting diagonal weights to 1... ")
            eye = np.expand_dims(np.eye(N),0)
            self.loss_weights = self.loss_weights * (1-eye) # zeroing diagonals
            self.loss_weights += eye # oneing diagonals
            print("torchiflying... ")
            self.loss_weights = torch.tensor(self.loss_weights).float()
            print("Loss weights computed")

        elif new_edge_scaling_method == 'none':

            print("Stacking... ")
            self.loss_weights = np.stack([g['adj_labels'].numpy() for g in self.data_list])
            self.loss_weights.fill(1)
            print("torchiflying... ")
            self.loss_weights = torch.tensor(self.loss_weights).float()
            print("Loss weights computed")

        self.val_test_data_gen()

    def val_test_data_gen(self):
        """Loop over the graph time sequence and compute a random set as a test set"""

        self.val_edges_list = []
        self.val_edges_false_list = []
        self.test_edges_list = []
        self.test_edges_false_list = []

        # new edges
        self.all_pos_edge_set = []
        self.new_edges_list = []
        self.new_edges_false_list = []

        # Loop over the sequence length. 
        # So if seq_len is 30, i will be 0...29 basically every graph in the time series
        for i in range(self.args.seq_len):

            val_test_graph, _ = load_adj_graph(f'{self.data_loc}_t{i}.npz')
            val_test_graph_adj, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(val_test_graph, test_percent=30., val_percent=20.)
            self.val_edges_list.append(val_edges)
            self.val_edges_false_list.append(val_edges_false)
            self.test_edges_list.append(test_edges)
            self.test_edges_false_list.append(test_edges_false)
            pos_edges = np.concatenate((val_edges, test_edges, train_edges)).tolist()
            self.all_pos_edge_set.append(set(map(tuple, pos_edges)))

        # Look over sequence again to get the new edges at each time point
        for i in range(self.args.seq_len):
            if i == 0: # on first loop do nothing
                self.new_edges_list.append(None)
                self.new_edges_false_list.append(None)

            # new edges since the last time step
            new_edges = np.array(list(self.all_pos_edge_set[i] - self.all_pos_edge_set[i-1]))
            if len(new_edges) == 0: # if the edge list is empty
                self.new_edges_list.append(None)
                self.new_edges_false_list.append(None)
            else:
                num_edges = len(new_edges)
                self.new_edges_list.append(new_edges)
                self.new_edges_false_list.append(self.test_edges_false_list[i][:num_edges])

        print("Validation and Test edges captured from last graph in the sequence")

        # Set the number of vertices in the graph
        self.num_nodes = val_test_graph.shape[0]

    def get_none_temporal_graph(self, temporal_id):
        """Get a certain graph as a none offset pair"""

        adj_train, features = load_adj_graph(f'{self.data_loc}_t{temporal_id}.npz')
        return prepare_data_for_model(adj_train, adj_train, "cpu")