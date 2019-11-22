import argparse
import csv
import datetime
import os
import time
from collections import defaultdict
from time import gmtime, strftime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.TNA import TNA_model
from utils import GraphData, evaluate_on_testset, run_test_set, save_model

torch.cuda.empty_cache()

def main(args):
    """ Train a TNA Model """ 

    # Compute the device upon which to run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")
    print("Training Temporal Neighbourhood Aggregation")
    print(f'Using {args.dataset} dataset')
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # TensorboardX logging
    time_cur = strftime("%d-%m-%Y_%H:%M", gmtime())
    writer = SummaryWriter(f"{args.logs_path}_{time_cur}")

    # Results
    results = defaultdict(list)
    
    # Set the number of timesteps in the sequence
    num_pairs = args.seq_len - 1 # one timestep per pair of consecutive graphs
    train_seq_length = num_pairs - 1 # Num training loops to actually do (keep last graph for test/validation)
    print(f'Requested sequence length: {args.seq_len}')
    print(f'Number of offset pairs in the sequence: {num_pairs}')

    # Load data
    data_loader = GraphData(args, num_pairs, new_edge_scaler=args.new_edge_scaler, new_edge_scaling_method=args.new_edge_scaling_method)
    num_nodes = data_loader.num_nodes # Set the number of vertices in the graph

    # Setup model and send to device
    tna = TNA_model(input_dim=num_nodes, n_hidden=args.hidden_size1, n_latent=args.hidden_size2, dropout=args.dropout, bias=args.gcn_bias, device=device, rnn_model=args.rnn_model, use_ln=args.use_ln, use_skip=args.use_skip)
    optimizer = optim.RMSprop(tna.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tna.to(device)
    tna.reset_hidden_states(device)
    tna.device = device
    print(f'Model has {sum([x.numel() for x in tna.parameters()])} parameters')

    # Full batch training loop ------------------------------------------------------------------------
    for epoch in range(args.num_epochs):
        
        # Reset
        epoch_loss = 0.
        epoch_training_acc = 0.

        # Loop over time
        for i in range(train_seq_length):

            # moved into time loop
            optimizer.zero_grad()

            # Load the current adjacency matrix and the next as a target
            data = data_loader.data_list[i]
            data['adj_norm'] = data['adj_norm'].to(device)
            data['adj_labels'] = data['adj_labels'].to(device)
            data['features'] = data['features'].to(device)

            # Update the sparsity weightings
            n_edges = (data['adj_labels'].sum())
            tna.pos_weight = float(num_nodes * num_nodes - n_edges) / n_edges
            tna.norm = float(num_nodes * num_nodes) / ((num_nodes * num_nodes - n_edges) * 2)
            tna.norm = torch.as_tensor(tna.norm).to(device)

            t = time.time()
        
            # forward pass
            tna.train()
            logits = tna(data['features'], data['adj_norm'])
            
            # Compute the VAE loss
            loss = tna.norm * F.binary_cross_entropy_with_logits(logits, data['adj_labels'], weight=data_loader.loss_weights[i].to(device), pos_weight=tna.pos_weight/4)
            kl = (0.5 / num_nodes) * torch.mean( torch.sum( 1 + 2 * tna.gc_var.log_sig - tna.gc_var.mu.pow(2) - torch.exp(tna.gc_var.log_sig).pow(2), 1) )
            loss -= kl
            epoch_loss += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Evaluation step on last graph in the sequence ----------------------------------------------------

        # Measure Training Accuracy 
        logits = torch.sigmoid(logits)
        correct_prediction = logits.ge(0.5).int().eq(data['adj_labels'].int())
        epoch_training_acc += torch.mean(correct_prediction.float())

        data = data_loader.data_list[-1] # Load graph seq_len - 2 as the input graph
        data['adj_norm'] = data['adj_norm'].to(device)
        data['features'] = data['features'].to(device)

        accuracy, roc_score, ap_score, tn, fp, fn, tp = evaluate_on_testset(tna, data_loader.val_edges_list[-1], data_loader.val_edges_false_list[-1], data)
        results['accuracy_train'].append(accuracy)
        results['roc_train'].append(roc_score)
        results['ap_train'].append(ap_score)

        print(f'Loss Value at Epoch {epoch+1}: {(epoch_loss/train_seq_length):.5f} Accuracy: {(epoch_training_acc/train_seq_length):.5f} val_acc= {accuracy:.5f} val_roc= {roc_score:.5f} vap_ap= {ap_score:.5f}')

        # Push data to visualisation stage ---------------------------------------------------------------
        writer.add_scalar('loss/train', (epoch_loss/train_seq_length), epoch)
        writer.add_scalar('acc/train', (epoch_training_acc/train_seq_length), epoch)
        writer.add_scalar('acc/val', accuracy, epoch)
        writer.add_scalar('auc/val', roc_score, epoch)
        writer.add_scalar('ap/val', ap_score, epoch)
        
        # Reset the hidden state for the next epoch
        tna.reset_hidden_states(device)

    print("Optimization Finished!")            
    run_test_set(tna, data_loader, device, train_seq_length)            

    # Save the model
    save_model(epoch, tna, optimizer, f'{args.dataset}_{args.num_epochs}_{args.rnn_model}.cpt')

if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='bitcoina', help='Dataset string.')
    parser.add_argument('--data_loc', type=str, default='/data/Temporal-Graph-Data/proc-data/', help='Dataset location string.')
    parser.add_argument('--seq_len', type=int, default=6, help='Length of the sequence to load.')  
    parser.add_argument('--logs_path', type=str, default='/tmp/tensorboard/', help='Where to log tensorboard.')  
    parser.add_argument('--test_freq', type=int, default=10, help='How often to run the test set.')
    parser.add_argument('--new_edge_scaling_method', type=str, default='none', help='Which method to use to weight new edges as they arrive in the graph.')
    parser.add_argument('--new_edge_scaler', type=float, default=50., help='How much to boost the weights on new edges.')

    parser.add_argument('--rnn_model', type=str, default='GRU',help='Which RNN to use.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_size1', type=int, default=32, help='Number of hidden units.')
    parser.add_argument('--hidden_size2', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--gcn_bias', type=int, default=0, help='If to use bias terms.')
    parser.add_argument('--use_features', type=str, default='n', help='Use vertex features or not.')                          
    parser.add_argument('--use_ln', type=int, default=1, help='If to use layer norm.')                           
    parser.add_argument('--use_skip', type=int, default=1, help='If to use a skip connection layer.')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
