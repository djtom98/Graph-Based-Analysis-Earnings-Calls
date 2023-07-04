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
from stellargraph import StellarDiGraph
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch
# import torch_geometric
# from torch_geometric.data import HeteroData
# import torch_geometric.transforms as T

import copy
#create one single graph from all the transcripts



from stellargraph.mapper import (
    CorruptedGenerator,
    FullBatchNodeGenerator,
    GraphSAGENodeGenerator,
    HinSAGENodeGenerator,
    ClusterNodeGenerator,
)
from stellargraph import StellarGraph
from stellargraph.layer import GCN, DeepGraphInfomax, GraphSAGE, GAT, APPNP, HinSAGE

from stellargraph import datasets
from stellargraph.utils import plot_history

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from IPython.display import display, HTML

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import Model

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

def build_edge_tensor_speaker(speakers):
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

def build_edge_tensor_text(texts,textsofsamespeaker,textposition):
    num_speakers = len(texts)
    edges = set()
    #adding edges between adjacent texts based on the order of the text in the textembeddings
    for textindices in textsofsamespeaker:
        for i in range(1, len(textindices)):
            prev_text = get_index_in_tensor(textindices[i - 1],textposition)
            current_text = get_index_in_tensor(textindices[i],textposition)
            edge = (prev_text, current_text)
            edges.add(edge)
    for i in range(1, num_speakers):
        prev_speaker = texts[i - 1]
        current_speaker = texts[i]
        edge = (prev_speaker, current_speaker)
        edges.add(edge)
    edges = list(edges)
    edge_tensor = torch.tensor(edges).T

    return edge_tensor

# %%
def create_heterograph(transc):    
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
    #preparing a list of text indices for the same speaker
    textsofsamespeaker=[]
    for i in range(len(transc.speakerunique.keys())):
        #get the text indices for each speaker
        textindices=transc.speakerunique[list(transc.speakerunique.keys())[i]].textindices
        textsofsamespeaker.append(textindices)
        for j in range(len(textindices)):
            speaker2text[0].append(i)
            speaker2text[1].append(get_index_in_tensor(textindices[j],textposition))
    speaker2text=torch.tensor(speaker2text)
    #speaker to speaker
    chunkspeakerlistasis=[]
    for i in range(0,len(transc.chunks),2):
        chunkspeakerlistasis.append([x[1] for x in list(speakerdict.values())].index(transc.chunks[i]))

    sp2sp_tensor = build_edge_tensor_speaker(chunkspeakerlistasis)
    # text2text
    txt2txt_tensor = build_edge_tensor_text(list(range(textembeddings.shape[0])),textsofsamespeaker,textposition)

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
    # data = T.ToUndirected()(data)
    return data

# %%
def build_edge_stellar_speaker(speakers):
    num_speakers = len(speakers)
    edges = set()

    for i in range(1, num_speakers):
        prev_speaker = speakers[i - 1]
        current_speaker = speakers[i]
        edge = (prev_speaker, current_speaker)
        edges.add(edge)
    edges = list(edges)
    # edge_tensor = torch.tensor(edges).T

    return edges

def build_edge_stellar_text(texts,textsofsamespeaker):
    num_speakers = len(texts)
    edges = set()
    #adding edges between adjacent texts based on the order of the text in the textembeddings
    for textindices in textsofsamespeaker:
        for i in range(1, len(textindices)):
            prev_text = textindices[i - 1]
            current_text = textindices[i]
            edge = (prev_text, current_text)
            edges.add(edge)
    for i in range(1, num_speakers):
        prev_speaker = texts[i - 1]
        current_speaker = texts[i]
        edge = (prev_speaker, current_speaker)
        edges.add(edge)
    edges = list(edges)
    # edge_tensor = torch.tensor(edges).T

    return edges
def create_hetero_stellar(transc):
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
    # textembeddings=torch.tensor(textembeddings)
    # textposition=torch.tensor(textposition)
        

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
    # speakerembeddings=torch.tensor(speakerembeddings)
    # speakerposition=torch.tensor(speakerposition)
        

    speaker2text=[[],[]]
    #preparing a list of text indices for the same speaker
    textsofsamespeaker=[]
    for i in transc.speakerunique.keys():
        #get the text indices for each speaker
        textindices=transc.speakerunique[i].textindices
        textsofsamespeaker.append(textindices)
        for j in textindices:
            speaker2text[0].append(i)
            speaker2text[1].append(j)
    # speaker2text=torch.tensor(speaker2text)
    # #speaker to speaker
    chunkspeakerlistasis=[x for x in transc.chunks if x in transc.speakerunique.keys()]
    sp2sp_edges = build_edge_stellar_speaker(chunkspeakerlistasis)
    # sp2sp_tensor = build_edge_tensor_speaker(chunkspeakerlistasis)
    # # text2text

    txt2txt_edges = build_edge_stellar_text(list(range(1,len(transc.chunks),2)),textsofsamespeaker)


    square_text = pd.DataFrame(textembeddings)
    square_text.index=textposition
    square_text.index=square_text.index.map(lambda x: "text"+str(x))

    square_speaker=pd.DataFrame(speakerposition)
    square_speaker.index=transc.speakerunique.keys()

    edges=pd.DataFrame(speaker2text).T
    edges['type']='sp2txt'
    edges.columns=['source','target','type']
    edges['target']=edges['target'].apply(lambda x: "text"+str(x))

    sp2sp=pd.DataFrame(sp2sp_edges)
    sp2sp['type']='sp2sp'
    sp2sp.columns=['source','target','type']
    txt2txt=pd.DataFrame(txt2txt_edges)
    txt2txt.columns=['source','target']
    txt2txt['source']=txt2txt['source'].apply(lambda x: "text"+str(x))
    txt2txt['target']=txt2txt['target'].apply(lambda x: "text"+str(x))
    txt2txt['type']='text2text'
    edges=edges.append(sp2sp)
    edges=edges.append(txt2txt)
    edges['source']=edges['source'].astype('str')
    edges['target']=edges['target'].astype('str')
    edges.reset_index(drop=True,inplace=True)

    

    square_everything_directed = StellarDiGraph(
        {"speaker": square_speaker, "text": square_text},
        edges,
        edge_type_column="type",
    )
    return square_everything_directed

# %%
cleanedec['stellar']=cleanedec.transcriptcls.apply(create_hetero_stellar)

# %%
len(cleanedec['transcriptcls'][0].chunks[5])
#find number of words
len(cleanedec['transcriptcls'][0].chunks[5].split())

# %%
modelgraphembeddings = SentenceTransformer('all-mpnet-base-v2')
# that's the sentence transformer
print(modelgraphembeddings.max_seq_length)
# that's the underlying transformer
print(modelgraphembeddings[0].auto_model.config.max_position_embeddings)

# %%
transc=cleanedec['transcriptcls'][5]
from collections import defaultdict

def feedtomodel(sentences):
    modelsent=[x[1] for x in sentences]
    modelsent=modelgraphembeddings.encode(modelsent)
    return [(t[0], modelsent[i]) for i, t in enumerate(sentences)]
def calculate_mean_tuples(tuples_list):
    result = defaultdict(list)
    for u, v in tuples_list:
        result[u].append(v)

    mean_tuples = ((u, np.mean(v_list, axis=0)) for u, v_list in result.items())
    return list(mean_tuples)

utterancelist=transc.chunks[1::2]
utter_long=[(i,x) for i,x in enumerate(utterancelist) if len(x.split())>=384]
#break each long utter into smaller utterances
# utter_longsplit=[(u,v.split()) for u,v in utter_long]
utter_longsplit=[(u," ".join(v[i:i+384])) for u,v in [(u,v.split()) for u,v in utter_long] for i in range(0,len(v),384)]
utter_longsplit=feedtomodel(utter_longsplit)
#take mean embedding for common indices 
utter_long=calculate_mean_tuples(utter_longsplit)
# utter_short=[(i,x) for i,x in enumerate(utterancelist) if len(x.split())<384]
# assert len(utterancelist)==len(utter_long)+len(utter_short)
# utter_long=feedtomodel(utter_long)
# utter_short=feedtomodel(utter_short)
# utter=utter_long+utter_short
# # utter.sort(key=lambda x: x[0])
# sorted(utter, key=lambda x: x[0])
# textembeddings=[u[1] for u in utter]





#merge la and lb based on the index
# l=la+lb
# sorted(l, key=lambda x: x[0])



# %%
[u[1].shape for u in utter_long]

# %%

for i in range(1,len(transc.chunks),2):
    #check length of text
    nwords=len(transc.chunks[i].split())
    if nwords>384:
        embtemp=[]
        #get embeddings of each 384 segment till the end and mean the embeddings
        embtemp.append(modelgraphembeddings.encode(transc.chunks[i][:384]))
        embtemp.append(modelgraphembeddings.encode(transc.chunks[i][384:]))
    else:
        modelgraphembeddings.encode(transc.chunks[i])
        break

# %%
#make individual graphs and then combine them by concatenating the pandas dataframes
modelgraphembeddings = SentenceTransformer('all-mpnet-base-v2')
from tqdm import tqdm
    # textdict={}
#define nodes of large graph
largesquare_text=pd.DataFrame()
largesquare_speaker=pd.DataFrame()
largeedges=pd.DataFrame()
for transcindex,transc in tqdm(enumerate(cleanedec.transcriptcls)):
    textembeddings=[]
    textposition=list(range(1,len(transc.chunks),2))
    # utteranceindex=0
    # for i in range(1,len(transc.chunks),2):
    #     textposition.append(i)
    #     #check length of text
    #     # if len(transc.chunks[i])>384:
    #     #     # n=
    #     #     embtemp=[]
    #     #     #get embeddings of each 384 segment till the end and mean the embeddings
            
    #     #     embtemp.append(modelgraphembeddings.encode(transc.chunks[i][:384]))
    #     #     embtemp.append(modelgraphembeddings.encode(transc.chunks[i][384:]))
    #     # else:
    #     textembeddings.append(modelgraphembeddings.encode(transc.chunks[i]))
        
    #     # textdict[utteranceindex]=i
    #     utteranceindex+=1
    utterancelist=transc.chunks[1::2]
    utter_long=[(i,x) for i,x in enumerate(utterancelist) if len(x.split())>=384]
    #break each long utter into smaller utterances
    # utter_longsplit=[(u,v.split()) for u,v in utter_long]
    utter_longsplit=[(u," ".join(v[i:i+384])) for u,v in [(u,v.split()) for u,v in utter_long] for i in range(0,len(v),384)]
    utter_longsplit=feedtomodel(utter_longsplit)
    #take mean embedding for common indices 
    utter_long=calculate_mean_tuples(utter_longsplit)
    utter_short=[(i,x) for i,x in enumerate(utterancelist) if len(x.split())<384]
    utter_short=feedtomodel(utter_short)
    
    utter=utter_long+utter_short
    assert len(utter)==len(utter_long)+len(utter_short)
    # utter.sort(key=lambda x: x[0])
    utter=sorted(utter, key=lambda x: x[0])
    textembeddings=[u[1] for u in utter]
        

    # get representations for each speaker
    # speakerdict={}
    # speakerembeddings=[]
    speakerposition=[]
    speakerindex=0
    for i in range(len(transc.speakerunique.keys())):
        #here you can create an embedding through different methods like from zero or orthogonal init or from a random walk of neighbours or lle
        # speakerembeddings.append(modelgraphembeddings.encode(list(transc.speakerunique.keys())[i]))
        # speakerdict[speakerindex]=(i,list(transc.speakerunique.keys())[i])
        speakerposition.append(i)
        speakerindex+=1
    # speakerembeddings=torch.tensor(speakerembeddings)
    # speakerposition=torch.tensor(speakerposition)
        

    speaker2text=[[],[]]
    #preparing a list of text indices for the same speaker
    textsofsamespeaker=[]
    for i in transc.speakerunique.keys():
        #get the text indices for each speaker
        textindices=transc.speakerunique[i].textindices
        textsofsamespeaker.append(textindices)
        for j in textindices:
            speaker2text[0].append(i)
            speaker2text[1].append(j)
    # speaker2text=torch.tensor(speaker2text)
    # #speaker to speaker
    chunkspeakerlistasis=[x for x in transc.chunks if x in transc.speakerunique.keys()]
    sp2sp_edges = build_edge_stellar_speaker(chunkspeakerlistasis)
    # sp2sp_tensor = build_edge_tensor_speaker(chunkspeakerlistasis)
    # # text2text

    txt2txt_edges = build_edge_stellar_text(list(range(1,len(transc.chunks),2)),textsofsamespeaker)


    square_text = pd.DataFrame(textembeddings)
    square_text.index=textposition
    square_text.index=square_text.index.map(lambda x: str(transcindex)+"_text"+str(x))
    largesquare_text=largesquare_text.append(square_text)

    square_speaker=pd.DataFrame(speakerposition)
    square_speaker['transcript']=transc.speakerunique.keys()
    square_speaker['transcript']=square_speaker.transcript.apply(lambda x: str(transcindex)+"_"+x)
    square_speaker.index=square_speaker.transcript
    #drop
    square_speaker.drop('transcript',axis=1,inplace=True)
    largesquare_speaker=largesquare_speaker.append(square_speaker)

    edges=pd.DataFrame(speaker2text).T
    edges['type']='sp2txt'
    edges.columns=['source','target','type']
    edges['target']=edges['target'].apply(lambda x:str(transcindex)+ "_text"+str(x))
    edges['source']=edges['source'].apply(lambda x:str(transcindex)+'_'+str(x))

    sp2sp=pd.DataFrame(sp2sp_edges)
    sp2sp['type']='sp2sp'
    sp2sp.columns=['source','target','type']
    sp2sp['source']=sp2sp['source'].apply(lambda x: str(transcindex)+"_"+str(x))
    sp2sp['target']=sp2sp['target'].apply(lambda x: str(transcindex)+"_"+str(x))
    
    txt2txt=pd.DataFrame(txt2txt_edges)
    txt2txt.columns=['source','target']
    txt2txt['source']=txt2txt['source'].apply(lambda x: str(transcindex)+"_text"+str(x))
    txt2txt['target']=txt2txt['target'].apply(lambda x: str(transcindex)+ "_text"+str(x))
    txt2txt['type']='text2text'
    edges=edges.append(sp2sp)
    edges=edges.append(txt2txt)
    edges['source']=edges['source'].astype('str')
    edges['target']=edges['target'].astype('str')
    edges.reset_index(drop=True,inplace=True)
    largeedges=largeedges.append(edges)
#117 min

# %%
largeedges.to_csv('../data/graph/largeedges_0107.csv')
largesquare_speaker.to_csv('../data/graph/largesquare_speaker_0107.csv')
largesquare_text.to_csv('../data/graph/largesquare_text_0107.csv')

# %%
largesquare_speaker

# %%
largesquare_text

# %%
largeedgesdupl=largeedges.reset_index(drop=True)
largeedgesdupl

# %%

G = StellarDiGraph(
    {"speaker": largesquare_speaker, "text": largesquare_text},
    largeedgesdupl,
    edge_type_column="type",
)

# %%
# # pickle the large graph\
# pickle.dump(G,open("../data/graph/largegraph_0107.pickle","wb"))

# %%
#pickle the data
# pickle.dump(cleanedec,open("../data/graph/stellar.pickle", "wb"))

# %%
#load the stellar df
cleanedec=pickle.load(open("../data/graph/stellar.pickle", "rb"))

# %%


# %%
cleanedec[(cleanedec.symbol=='NVS')&(cleanedec.year==2018)&(cleanedec.quarter==3)]

# %%
#visualization of a transcript in stellar
#let's take one transcript
import random
seed = random.randint(0, 2**32 - 1)
random.seed(seed)
np.random.seed(seed)
# g=cleanedec['stellar'][328].to_networkx()
g=cleanedec['stellar'][346].to_networkx()
filter=[x for x in list(g.nodes)if 'text' not in x]

# print(labels)
# alllayout=nx.bipartite_layout(g,nodes=filter)
# nx.draw_networkx_labels(g,labels)
# plt.show()

# layout=nx.spectral_layout(g)
# layout=nx.kamada_kawai_layout(g)
# layout=nx.spring_layout(g,scale=0.5,k=2)
layout=nx.spring_layout(g,scale=0.05,k=2)
nx.draw(g.subgraph(filter),with_labels=True,pos=layout,font_size=8,node_size=4000,node_color='white')
#draw nodes with just the labels
# nx.draw_networkx_labels(g.subgraph(filter),pos=layout,font_size=10)
plt.show()

# %%


# %%

pos = nx.nx_agraph.graphviz_layout(g, prog="twopi", args="")
plt.figure(figsize=(8, 8))
colorfilter=[]
sizefilter=[]
labels = {}    
for node in g.nodes():
    if node in [x for x in list(g.nodes)if 'text' not in x]:
        #set the node name as the key and the label as its value 
        labels[node] = node
for node in g.nodes():
    if 'text' in node:
        colorfilter.append('red')
        sizefilter.append(20)
    else:
        colorfilter.append('darkblue')
        sizefilter.append(150)
nx.draw(g, pos, node_size=sizefilter, alpha=0.8, node_color=colorfilter, with_labels=False)
# nx.draw_networkx_labels(g,pos,labels,font_size=10,font_color='black')
# import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
bcircle = Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=7, label='Text',alpha=0.8)
rcircle = Line2D([], [], color='darkblue', marker='o', linestyle='None',
                          markersize=7, label='Speaker',alpha=0.8)
custom_lines =[bcircle,rcircle]
plt.legend(handles=custom_lines)
plt.axis("equal")
plt.tight_layout()
plt.show()

# %%
# load the graph
G=pickle.load(open("../data/graph/largegraph_0107.pickle","rb"))

# %%
print(G.info())

# %%
# len(largesquare_speaker.index)

# %%
# filter=largesquare_speaker[largesquare_speaker[0].isin([0,1,2])].index
# filter=largesquare_speaker.index
filter=list(G.nodes(node_type="speaker"))
#let us create several test_gen of size batch size so that we get an embedding for each node

test_gen=hinsage_generator.flow(filter)

# %%
y=pd.DataFrame({'speakername':filter})
y['transcriptid']=y['speakername'].apply(lambda x: int(x.split("_")[0]))
#group by transcriptid give row number
y['speakerid']=y.groupby('transcriptid').cumcount()
#if speakerid is 0,1,2 then 1 else 0
y['label']=y['speakerid'].apply(lambda x: 1 if x in [1] else 0)
y=y.merge(cleanedec.symbol, left_on='transcriptid', right_index=True)
y['embeddings']=graphsageembs

# %%
y[y.speakerid.isin([1,2,3,4])].speakername.unique().tolist()

# %%
y.head()

# %%
y

# %%
#preparing the cosine similarities
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.special import kl_div

# Assuming your dataframe is called 'df'

# Group the dataframe by 'documentid'
grouped = y.groupby('transcriptid')

# Initialize an empty list to store the average cosine similarities
avg_cos_similarities = []
max_cos_similarities=[]
min_cos_similarities=[]
avg_eucl_dist=[]
max_eucl_dist=[]
min_eucl_dist=[]
avg_kl_div=[]

# Iterate over each group (document)
for documentid, group in grouped:
    # Get the speaker embedding for the speaker from the company
    # company_speaker_embedding = group[group['label'] == 1]['embeddings'].values[0]
    company_speaker_embedding=group[group['speakerid'].isin([1,2,3])]['embeddings'].mean(axis=0)

    # Calculate the threshold value for speaker ID filtering
    threshold = 0.3 * (group['speakerid'].max() - group['speakerid'].min())
    
    # Get the sampled analyst embeddings
    analyst_embeddings = group[(group['label'] == 0) & (group['speakerid'] > threshold)]['embeddings'].sample(5,replace=True).values
    #reshaping
    sample_cosine_similarities=[]
    sample_euclidean_dist=[]
    sample_kl_divergence=[]
    for analyst in analyst_embeddings:
        ac=cosine_similarity(company_speaker_embedding.reshape(1,-1), analyst.reshape(1,-1))
        euclidean_distance = euclidean(company_speaker_embedding, analyst)
        # kl_divergence = kl_div(company_speaker_embedding+1e-8, analyst+1e-8).sum()
        kl_divergence = kl_div(company_speaker_embedding, analyst).sum()

        sample_cosine_similarities.append(ac)
        sample_euclidean_dist.append(euclidean_distance)
        sample_kl_divergence.append(kl_divergence)

    sample_cosine_similarities=np.array(sample_cosine_similarities).reshape(-1,1).flatten()
    sample_euclidean_dist=np.array(sample_euclidean_dist).reshape(-1,1).flatten()
    sample_kl_divergence=np.array(sample_kl_divergence).reshape(-1,1).flatten()
    # # Compute the cosine similarity between the company speaker and sampled analysts
    # cosine_similarities = cosine_similarity(company_speaker_embedding.reshape(-1,1), analyst_embeddings.reshape(-1,1))
    
    # Calculate the average cosine similarity
    cos_similarity = np.mean(sample_cosine_similarities)
    cos_similarity_max = np.max(sample_cosine_similarities)
    cos_similarity_min = np.min(sample_cosine_similarities)
    eucl_dist=np.mean(sample_euclidean_dist)
    eucl_dist_max=np.max(sample_euclidean_dist)
    eucl_dist_min=np.min(sample_euclidean_dist)
    kl_divergence=np.mean(sample_kl_divergence)
    
    # Append the average cosine similarity to the list
    avg_cos_similarities.append(cos_similarity)
    max_cos_similarities.append(cos_similarity_max)
    min_cos_similarities.append(cos_similarity_min)
    avg_eucl_dist.append(eucl_dist)
    max_eucl_dist.append(eucl_dist_max)
    min_eucl_dist.append(eucl_dist_min)
    avg_kl_div.append(kl_divergence)

# Add the average cosine similarities to a new column in the dataframe
df=pd.DataFrame()
df['avg_cosine_similarity_company_analyst'] = avg_cos_similarities
df['max_cosine_similarity_company_analyst'] = max_cos_similarities
df['min_cosine_similarity_company_analyst'] = min_cos_similarities
df['avg_euclidean_distance'] = avg_eucl_dist
df['max_euclidean_distance'] = max_eucl_dist
df['min_euclidean_distance'] = min_eucl_dist
df['avg_kl_divergence'] = avg_kl_div

# %%
#create an outdf with y rows only having value 1
outdf=y[y['speakerid'].isin([1,2,3])]
outdf.index=outdf.transcriptid
#rename index
outdf.index.name='id'
outdf=outdf.groupby(['transcriptid','symbol']).agg({'speakername':'sum','speakerid':'max','speakerid':'max','embeddings':'mean','label':'max'})
outdf.reset_index(inplace=True)
# #merge df with outdf
outdf=outdf.merge(df,left_index=True,right_index=True)
# #merge outdf with cleaned ec
outdf=outdf.merge(cleanedec,left_index=True,right_index=True)

# %%
outdf.to_csv('../data/graph/graphfeatures_10_7_0107.csv')

# %%
y1=y['label'].to_list()
y2=y['symbol'].to_list()

# %%
trans = TSNE(n_components=2)
emb_transformed = pd.DataFrame(trans.fit_transform(np.array(graphsageembs)))
emb_transformed["label"] = y1
emb_transformed["symbol"] = y2
# convert dtype to categorical
emb_transformed["symbol"] = pd.Categorical(emb_transformed["symbol"])
emb_transformed['transcriptid']=y['transcriptid']
emb_transformed['speakerid']=y['speakerid']
#convert categorical to numeric
# emb_transformed["symbolnum"] = emb_transformed["symbol"].cat.codes

# %%
emb_transformed.symbol.cat.categories.get_loc('NVS')
emb_transformed.symbol.cat.categories.tolist()
# sorted(emb_transformed.symbol.cat.codes.unique().tolist())

# %%
#graphfilter
#filter for just 4 companies
graph_df=emb_transformed.copy()
# graph_df=emb_transformed[emb_transformed['symbol'].isin([ 'MRK', 'BMY', 'NVS', 'PFE'])]
#within one earnings call
# graph_df=emb_transformed[emb_transformed['transcriptid']==40]
#stratify sample on the label 
graph_df=graph_df.groupby(['label','symbol']).apply(lambda x: x.sample(frac=0.1))

# graph_df=emb_transformed[emb_transformed['label']==0].sample(500)
# graph_df=emb_transformed.iloc[2000:2500]
# graph_df=emb_transformed[emb_transformed['label']==1]
graph_df["symbol"] = pd.Categorical(graph_df["symbol"])

# %%
alpha = 0.7
# norm = mpl.colors.Normalize(vmin=emb_transformed.symbol.min(), vmax=emb_transformed.symbol.max())
# cmap = plt.colormaps["plasma"]
def marker_style(label):
    if label == 1:
        return "^"
    else:
        return "o"
def ret_col(symbol):
    slist=['ABBV', 'AZN', 'BMY', 'JNJ', 'LLY', 'MRK', 'NVO', 'NVS', 'PFE', 'ROG']
    clist=['red','blue','green','c','pink','orange','purple','black','brown','grey']
    #zip into dict
    clist=dict(zip(slist,clist))
    return clist[symbol]
cmap = plt.get_cmap("coolwarm")
fig, ax = plt.subplots(figsize=(7, 7))
data = zip(graph_df[0],graph_df[1],graph_df["label"],graph_df["symbol"])
for x1, x2, label, symbol in data:
    m = marker_style(label)
    ms = None if m == "o" else 12
    # ax.plot(x1,x2, marker=m, color=cmap(norm(symbol)),alpha=alpha)
    ax.plot(x1,x2, marker=m, color=ret_col(symbol),alpha=alpha,markersize=ms)
# ax.scatter(
#     emb_transformed[0],
#     emb_transformed[1],
#     marker=emb_transformed["label"],
#     c=emb_transformed['ec_id'],
#     cmap="jet",
#     alpha=alpha,
# )
ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
plt.title("TSNE visualization of GraphSAGE embeddings for speaker nodes")
#legend for the symbols color
# import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# custom_lines = [Patch(facecolor=ret_col(x),linewidth=0.5) for x in sorted(graph_df.symbol.cat.codes.unique().tolist())]
custom_lines = [Line2D([0], [0], color=ret_col(x), lw=2,label=x) for x in sorted(graph_df.symbol.unique().tolist())]
# ax.legend(custom_lines, sorted(graph_df.symbol.unique().tolist()))

#legend for the shapes, triangle is the company speaker and circle is the analyst
triangle = Line2D([], [], color='gray', marker='^', linestyle='None',
                          markersize=10, label='Company')
circle = Line2D([], [], color='gray', marker='o', linestyle='None',
                          markersize=10, label='Analyst')
custom_lines +=[triangle,circle]
ax.legend(handles=custom_lines)
plt.show()

# %%
alpha = 0.7

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(
    graph_df[0],
    graph_df[1],
    c=graph_df["label"],
    cmap="viridis",
    alpha=alpha,
)
ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
#legend for the label
bcircle = Line2D([], [], color='yellow', marker='o', linestyle='None',
                          markersize=7, label='Company')
rcircle = Line2D([], [], color='purple', marker='o', linestyle='None',
                          markersize=7, label='Analyst')
custom_lines =[bcircle,rcircle]
ax.legend(handles=custom_lines)
plt.title("TSNE visualization of GraphSAGE embeddings for speaker nodes")
plt.show()

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
# graphtodisplay=copy.deepcopy(cleanedec['graphobj'][1])
# visualize_graph(graphtodisplay)

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
#creates and saves a graph dataset with the graph objects to the data/graph folder
graphdata=ECgraphDataset(data_list=cleanedec['graphobj'].tolist())


