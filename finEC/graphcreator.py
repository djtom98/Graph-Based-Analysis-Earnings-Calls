# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: thesis
#     language: python
#     name: python3
# ---

# %%
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


# %%
#get the data
ec=pickle.load(open("../data/Sentiment_Market_Pharma/earnings_call_top10_ph.pickle", "rb"))
ec=ec.reset_index()

# %%
cleanedec=dpp.process_ec(ec,0)

# %%
# re.findall(,cleanedec.cleanedec[0])
# "(?i)(?:[. ]{1,2})([ a-zA-Z]*)(?::)"gm
pattern=r'(?i)(?:[. ]{0,2})((\b\w+\b[\s\r\n]*){1,3})(?::)'
# Then use the following to get all overlapping indices:
input=cleanedec.cleanedec[0]
indicesTuple = [(mObj.start(1),mObj.end(1)) for mObj in re.finditer(pattern,input)]


# %%
class speaker(object):
    newid = itertools.count().__next__
    instances = weakref.WeakValueDictionary()
    
    def __init__(self,name,indices):
        self.id = speaker.newid()
        self.name=name 
        self.indices=indices
    def addtext(self,textindices,text):
        self.text=text
        self.textindices=textindices
    #return the 
    
class Transcript():
    def __init__(self,ec):
        self.speakers=[]
        self.speakerunique=dict()
        # self.speakerindices=self.get_all_speakerindices(ec)
        self.cleanedtext=ec
        self.chunks=self.break_into_chunks()
        self.embed_speakers()

    def get_valid_speakers(self):
        pass
        
    def get_all_speakerindices(self,ec):
        pattern=r"(?i)(?:[.\s\r\n]{0,2})((\b\w+\b[\s\r\n]*){1,3})(?::)"
        
        # pattern=r"(?i)(?:[.]){1}(?:[\s\r\n]{0,2})((\b\w+\b[\s\r\n]*){1,3})(?::)"
        indicesTuple = [(mObj.start(1),mObj.end(1)) for mObj in re.finditer(pattern,ec)]
        return indicesTuple
    
    def break_into_chunks(self):
        speakerindices=self.get_all_speakerindices(self.cleanedtext)
        broken=[]
        for i in range(len(speakerindices)-1):
            broken.append(self.cleanedtext[speakerindices[i][0]:speakerindices[i][1]])
            broken.append(self.cleanedtext[speakerindices[i][1]+1:speakerindices[i+1][0]])
        return broken
        # self.cleanedtext 
    def get_speakers(self):
        for i in range(len(self.speakerindices)-1):
            self.speakers.append(speaker(self.cleanedtext[self.speakerindices[i][0]:self.speakerindices[i][1]],self.speakerindices[i]))
    
    def embed_speakers(self):
        speakerindices=dict()
        for i in range(0,len(self.chunks),2):
            if self.chunks[i] not in speakerindices.keys():
                speakerindices[self.chunks[i]]=[]
            speakerindices[self.chunks[i]].append(i)
        for sp in speakerindices.keys():
            self.speakerunique[sp]=speaker(sp,speakerindices[sp])
            textlist=[]
            textindices=[]
            for index in speakerindices[sp]:
                textlist.append(self.chunks[index+1])
                textindices.append(index+1)
            self.speakerunique[sp].addtext(textindices,textlist)
            self.speakers.append(sp)

'''improvements to be made: add levenshtein distance to check if there are similar speakers and collate them into one 
import nltk
from nltk.corpus import stopwords

def preprocess_string(input_string):
    # Convert to lowercase and remove punctuation
    processed_string = ''.join(c.lower() for c in input_string if c.isalnum() or c.isspace())
    
    # Remove common stop words
    stop_words = set(stopwords.words('english'))
    processed_string = ' '.join(word for word in processed_string.split() if word not in stop_words)
    
    return processed_string

def calculate_similarity(str1, str2):
    # Preprocess both strings
    str1 = preprocess_string(str1)
    str2 = preprocess_string(str2)
    
    # Calculate Levenshtein distance
    distance = nltk.edit_distance(str1, str2)
    
    # Calculate similarity score
    max_length = max(len(str1), len(str2))
    similarity = (max_length - distance) / max_length
    
    return similarity

def compare_names(name1, name2):
    similarity_threshold = 0.8  # Adjust this threshold based on your requirements
    
    similarity_score = calculate_similarity(name1, name2)
    
    if similarity_score >= similarity_threshold:
        return True  # Names are similar or the same person
    else:
        return False  # Names are different people
'''

# %%
cleanedec['transcriptcls']=cleanedec.cleanedec.apply(lambda x: Transcript(x))


# %%
def get_index_in_tensor(i,textposition):
    return torch.where(textposition==i)[0][0].item()

def build_edge_tensor(speakers):
    num_speakers = len(speakers)
    edges = set()

    for i in range(1, num_speakers):
        prev_speaker = speakers[i - 1]
        current_speaker = speakers[i]
        edge = (prev_speaker, current_speaker)
        edges.add(edge)
    edges = list(edges)
    edge_tensor = torch.tensor(edges).T

    return edge_tensor

#creating a function that wraps the creation of a heterograph from the transcript class
def create_heterograph(transc):
#make a graph with the speakers as nodes and the text as another type of node
#then make the edges between the speakers and the text
# get embeddings for each text
    modelgraphembeddings = SentenceTransformer('all-mpnet-base-v2')
    # textdict={}
    textembeddings=[]
    textposition=[]
    utteranceindex=0
    for i in range(1,len(transc.chunks),2):
        textposition.append(i)
        textembeddings.append(modelgraphembeddings.encode(transc.chunks[i]))
        # textdict[utteranceindex]=i
        utteranceindex+=1
    textembeddings=torch.tensor(textembeddings)
    textposition=torch.tensor(textposition)
        
    # get representations for each speaker
    speakerdict={}
    speakerembeddings=[]
    speakerposition=[]
    speakerindex=0
    for i in range(len(transc.speakerunique.keys())):
        #here you can create an embedding through different methods like from zero or orthogonal init or from a random walk of neighbours or lle
        speakerembeddings.append(modelgraphembeddings.encode(list(transc.speakerunique.keys())[i]))
        speakerdict[speakerindex]=(i,list(transc.speakerunique.keys())[i])
        speakerposition.append(i)
        speakerindex+=1
    speakerembeddings=torch.tensor(speakerembeddings)
    speakerposition=torch.tensor(speakerposition)
        
    speaker2text=[[],[]]
    for i in range(len(transc.speakerunique.keys())):
        #get the text indices for each speaker
        textindices=transc.speakerunique[list(transc.speakerunique.keys())[i]].textindices
        for j in range(len(textindices)):
            speaker2text[0].append(i)
            speaker2text[1].append(get_index_in_tensor(textindices[j],textposition))
    speaker2text=torch.tensor(speaker2text)
    #speaker to speaker
    chunkspeakerlistasis=[]
    for i in range(0,len(transc.chunks),2):
        chunkspeakerlistasis.append([x[1] for x in list(speakerdict.values())].index(transc.chunks[i]))

    sp2sp_tensor = build_edge_tensor(chunkspeakerlistasis)
    # text2text
    txt2txt_tensor = build_edge_tensor(list(range(textembeddings.shape[0])))

    # build graph
    data = HeteroData()
    data['text'].x = textembeddings
    data['text'].pos = textposition
    data['speaker'].x = speakerembeddings
    data['speaker'].pos = speakerposition
    #building edges
    data['speaker', 'text'].edge_index = speaker2text
    data['speaker', 'speaker'].edge_index = sp2sp_tensor
    data['text', 'text'].edge_index = txt2txt_tensor

    #transformation
    data = T.ToUndirected()(data)
    # data = T.AddSelfLoops()(data)
    return data
            


# %%
cleanedec['graphobj']=cleanedec.transcriptcls.apply(create_heterograph)


# %%
def visualize_graph(graphtodisplay):
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



# %%
type(graphtodisplay)

# %%
graphtodisplay=copy.deepcopy(cleanedec['graphobj'][1])
visualize_graph(graphtodisplay)

# %%
# get all graph objects into a list
cleanedec['graphobj'].tolist()

# %%
bool(['s'])

# %%
#convolution on the graph to update the embeddings of the speaker nodes
#creating a custom dataset for convenience
import torch
from torch_geometric.data import InMemoryDataset, download_url


class ECgraphDataset(InMemoryDataset):
    def __init__(self, root='../data/graph', transform=None, pre_transform=None, pre_filter=None,data_list=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # @property
    # def raw_file_names(self):
    #     return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     url = 'https://some_url.com'
    #     download_url(url, self.raw_dir)

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])


# %%
graphdata=ECgraphDataset(data_list=cleanedec['graphobj'].tolist())

# %%
graphdatatest=ECgraphDataset()

# %%
graphdatatest[1]==graphdata[1]

# %%
graphdatatest.data

# %%
graphdatatest.slices

# %%
