import pandas as pd
import numpy as np
import pickle
import finEC.datapreproc as dpp
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import warnings
warnings.filterwarnings('ignore')
import regex as re
import weakref
import itertools

from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch_geometric
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

import copy



def visualize_graph(graphtodisplay):
    """function to visualize a graph"""  
    graph = nx.MultiDiGraph()  # Create a directed graph that allows multiple edges between nodes
    # Add 'speaker' nodes with 'type' attribute set to 'speaker'
    speakers = list(range(graphtodisplay['speaker'].num_nodes))
    graph.add_nodes_from(speakers, node_type='speaker')
    delta=graphtodisplay['speaker'].num_nodes

    # Add 'text' nodes with 'type' attribute set to 'text'
    texts = list(range(delta,graphtodisplay['text'].num_nodes+delta))
    graph.add_nodes_from(texts, node_type='text')

    # Add edges between 'speaker' and 'text' nodes
    speaker_text_edges = [tuple([x[0],x[1]+delta]) for x in graphtodisplay['speaker','text'].edge_index.T.tolist()]
    graph.add_edges_from(speaker_text_edges, edge_type='speaker_text')

    # Add edges between 'speaker' nodes
    speaker_speaker_edges = [tuple([x[0],x[1]]) for x in graphtodisplay['speaker','speaker'].edge_index.T.tolist()]
    graph.add_edges_from(speaker_speaker_edges, edge_type='speaker_speaker')

    # Add edges between 'text' nodes
    text_text_edges = [tuple([x[0]+delta,x[1]+delta]) for x in graphtodisplay['text','text'].edge_index.T.tolist()]
    graph.add_edges_from(text_text_edges, edge_type='text_text')

    # Define node colors
    node_colors = {'speaker': 'red', 'text': 'blue'}

    # Create the plot
    plt.figure(figsize=(8, 8))
    pos = nx.kamada_kawai_layout(graph)  # Positions of nodes

    # Draw 'speaker' nodes with red color
    speaker_nodes = [node for node, data in graph.nodes(data=True) if data['node_type'] == 'speaker']
    nx.draw_networkx_nodes(graph, pos, nodelist=speaker_nodes, node_color=node_colors['speaker'], node_size=40)

    # Draw 'text' nodes with blue color
    text_nodes = [node for node, data in graph.nodes(data=True) if data['node_type'] == 'text']
    nx.draw_networkx_nodes(graph, pos, nodelist=text_nodes, node_color=node_colors['text'], node_size=10)

    # Draw edges
    nx.draw_networkx_edges(graph, pos)

    # Display the plot
    plt.axis('off')
    plt.show()

    #   #display this graph
    edgeindextest=graphtodisplay['speaker', 'speaker'].edge_index
    num_nodes = edgeindextest.max().item() + 1

    # Create a PyTorch Geometric Data object
    data = torch_geometric.data.Data(edge_index=edgeindextest)

    # Convert to NetworkX graph
    graph = torch_geometric.utils.to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=True)
    # Create the plot
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos, with_labels=True, node_size=1000, font_size=12, node_color='lightblue')

    # Display the plot
    plt.axis('off')
    plt.show()
