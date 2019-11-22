import csv
import datetime
import os
import pickle as pkl
import argparse

import networkx as nx
import numpy as np
import scipy.sparse as sp
from graph_tool.all import *
from sklearn import preprocessing

def load_adj_graph(data_location):
    """Load a given graph and return the adjacency matrix"""

    adj = sp.load_npz(data_location)
    return adj, None

def preprocess_adj(adj):
    """Preprocess the adjacency matrix and return"""

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    return adj

def save_gt_graph(graph, filename):
    """Save graph_tool graph as npz file"""

    adj = adjacency(graph)
    sp.save_npz(filename, adj)

def process_empirical_temporal_graphs(args, extract_features=False):
    """Function to load and process empirical temporal graphs"""
    
    # Load the edges from file
    data_location = os.path.join(args.org_data_path, args.edges_filepath)
    if args.weighted == 1:
        full_graph = load_graph_from_csv(data_location, directed=False, hashed=True, eprop_types=['int', 'int'], eprop_names=['w', 'date'], csv_options={'delimiter': args.dem_lim_str})
    else:
        full_graph = load_graph_from_csv(data_location, directed=False, hashed=True, eprop_types=['int'], eprop_names=['date'], csv_options={'delimiter': args.dem_lim_str})
    remove_self_loops(full_graph) # Remove any self loops
    print("Graph Loaded From Disk")
    print(full_graph)

    # Get the range of dates from the edges
    dates = np.array([full_graph.ep.date[e] for e in full_graph.edges()])

    # Here we can control the start point of the graph
    if args.min_date != 0:
        dates_min = args.min_date
    else:
        dates_min = dates.min()

    num_months = int((dates.max()- dates_min)* 3.8026E-7)
    num_weeks = int((dates.max()- dates_min)* 1.6534392E-6)
    num_days = int((dates.max()- dates_min)* 1.1574074E-5)
    print(f"First Edge Appeared At: {datetime.datetime.fromtimestamp(dates.min()).strftime('%c')}, we are processing from: {datetime.datetime.fromtimestamp(dates_min).strftime('%c')} ")
    print(f"Last Edge Appeared At: {datetime.datetime.fromtimestamp(dates.max()).strftime('%c')}")
    print(f'Data covers {num_months} Months, {num_weeks} Weeks and {num_days} Days')
    if args.granularity == 'm':
        time_range = num_months
    elif args.granularity == 'w':
        time_range = num_weeks
    elif args.granularity == 'd':
        time_range = num_days
    date_ranges = np.linspace(dates.min(), dates.max(), num=time_range+1) # Splits the timestamps into linearly separated ranges. 
    print("Timestamps Loaded")

    # Filter the graph
    save_path = os.path.join(args.new_data_path, args.data_save_str)
    graph_counter = 0
    time_lower_limit = 0
    edge_lookback = 15
    num_previous_edge_changes = 0.
    edge_change_list = []

    for i in range(1, len(date_ranges)):

        temp_graph = Graph(GraphView(full_graph, efilt=lambda e: full_graph.ep.date[e] <= int(date_ranges[i])), directed=False)

        remove_parallel_edges(temp_graph) # Remove any parallel edges
        assert full_graph.num_vertices() == temp_graph.num_vertices() # Check both graphs have the required number of vertices
        save_gt_graph(temp_graph, f'{save_path}_t{graph_counter}.npz')
        graph_counter += 1  

        print(temp_graph)
        print(f"Processing edges until: {datetime.datetime.fromtimestamp(int(date_ranges[i])).strftime('%c')} Epoch time: {date_ranges[i]}")

        if i != 0:
            edge_change_list.append(temp_graph.num_edges() - num_previous_edge_changes)
            num_previous_edge_changes = temp_graph.num_edges()

    print(f'Mean number of edges changes = {np.mean(edge_change_list)} +/- {np.std(edge_change_list)}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--org_data_path', type=str, default='/data/Temporal-Graph-Data/org-data/', help='Path to original data.')  
    parser.add_argument('--new_data_path', type=str, default='/data/Temporal-Graph-Data/proc-data/', help='Path to new data.')  
    parser.add_argument('--edges_filepath', type=str, default='out.ca-cit-HepTh', help='Dataset string.')
    parser.add_argument('--dem_lim_str', type=str, default='\t', help='Delimit string for the edge list.')
    parser.add_argument('--data_save_str', type=str, default='', help='Name of new dataset.')
    parser.add_argument('--granularity', type=str, default='m', help='month, week or year.')
    parser.add_argument('--num_time_splits', type=int, default=30, help='Number of splits to make.')
    parser.add_argument('--weighted', type=int, default=1, help='If the graph is weighted 1=weighted.')
    parser.add_argument('--min_date', type=int, default=0, help='The smallest timestamp from which to start 1=weighted.')

    args = parser.parse_args()

    process_empirical_temporal_graphs(args)
