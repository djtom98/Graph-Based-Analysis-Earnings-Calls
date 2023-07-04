# %%
import pandas as pd
import numpy as np
import pickle
import finEC.datapreproc as dpp
from graphutils import *
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

# %%
#load the stellar df
cleanedec=pickle.load(open("../data/graph/stellar.pickle", "rb"))
# load the graph
G=pickle.load(open("../data/graph/largegraph_0107.pickle","rb"))
print(G.info())
#create one single graph from all the transcripts

# %%
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

hinsage_generator = HinSAGENodeGenerator(
    G, batch_size=500, num_samples=[7,5], head_node_type="speaker",
)
#layer size 768 [5,3] works
hinsage_model = HinSAGE(
    layer_sizes=[768,768], generator=hinsage_generator
)
corrupted_generator = CorruptedGenerator(hinsage_generator)
gen = corrupted_generator.flow(G.nodes(node_type="speaker"))
infomax = DeepGraphInfomax(hinsage_model, corrupted_generator)
x_in, x_out = infomax.in_out_tensors()

# %%
model = Model(inputs=x_in, outputs=x_out)
model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=1e-3))
epochs = 100
es = EarlyStopping(monitor="loss", min_delta=0, patience=10)
history = model.fit(gen, epochs=epochs, verbose=0, callbacks=[es])
plot_history(history)
x_emb_in, x_emb_out = hinsage_model.in_out_tensors()
# for full batch models, squeeze out the batch dim (which is 1)
# x_out = tf.squeeze(x_emb_out, axis=0)
emb_model = Model(inputs=x_emb_in, outputs=x_emb_out)
print(len(list(G.nodes(node_type="speaker"))))
# len(largesquare_speaker.index)
# filter=largesquare_speaker[largesquare_speaker[0].isin([0,1,2])].index
# filter=largesquare_speaker.index
filter=list(G.nodes(node_type="speaker"))
#let us create several test_gen of size batch size so that we get an embedding for each node

# %%
#save emb_model
emb_model.save("../models/graph/emb_model_10_7.h5")

# %%
#load the model
import tensorflow_hub as hub
# emb_model=tf.keras.models.load_model("../models/graph/emb_model_10_7.h5")
emb_model = tf.keras.models.load_model(
       ("../models/graph/emb_model_10_7.h5"),
       custom_objects={'KerasLayer':hub.KerasLayer}
)

# %%
emb_model.summary()

# %%
test_gen=hinsage_generator.flow(filter)
graphsageembs=[]
for batch in range(0,test_gen.data_size//test_gen.batch_size+1):
    embeddings= emb_model.predict(test_gen[batch][0])
    graphsageembs+=[*embeddings]
y=pd.DataFrame({'speakername':filter})
y['transcriptid']=y['speakername'].apply(lambda x: int(x.split("_")[0]))
#group by transcriptid give row number
y['speakerid']=y.groupby('transcriptid').cumcount()
#if speakerid is 0,1,2 then 1 else 0
y['label']=y['speakerid'].apply(lambda x: 1 if x in [1] else 0)
y=y.merge(cleanedec.symbol, left_on='transcriptid', right_index=True)
y['embeddings']=graphsageembs
y[y.speakerid.isin([1,2,3,4])].speakername.unique().tolist()
y.head()

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

df.describe()

# %%
df.plot()
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
outdf
# y[(y['speakerid'].isin([1,2,3]))&(y.transcriptid==0)].embeddings.mean(axis=0)
# outdf[outdf['transcriptid']==0]
outdf.to_csv('../data/graph/graphfeatures_10_7_0107_02.csv')



# %%
# outdf=pd.read_csv('../data/graph/graphfeatures_10_7_0107_02.csv')

# %%
# checkthisobs=(outdf.quarter==4)&(outdf.year==2013)&(outdf.symbol_x=='NVS')
# obs=outdf[checkthisobs]
# transcriptclass=cleanedec[checkthisobs].transcriptcls
# obs

# %%
company=list(transcriptclass.values[0].speakerunique.keys())[1:4]
analysts=list(transcriptclass.values[0].speakerunique.keys())[int(round(0.3*len(list(transcriptclass.values[0].speakerunique.keys())),0)):]

# %%
companytext=[]
analytext=[]
for analyst in analysts:
    # print(analyst)
    analytext.append(transcriptclass.values[0].speakerunique[analyst].text)

for compguy in company:
    # print(compguy)
    companytext.append(transcriptclass.values[0].speakerunique[compguy].text)

# for textlist in transcriptclass.values[0].speakerunique['joseph jimenez'].text

# %%
companytext=' '.join([' '.join(x) for x in companytext])
analytext=' '.join([' '.join(x) for x in analytext])

# %%

y1=y['label'].to_list()
y2=y['symbol'].to_list()
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
emb_transformed.symbol.cat.categories.get_loc('NVS')
emb_transformed.symbol.cat.categories.tolist()
# sorted(emb_transformed.symbol.cat.codes.unique().tolist())
#graphfilter
#filter for just 4 companies
graph_df=emb_transformed.copy()
# graph_df=emb_transformed[emb_transformed['symbol'].isin([ 'MRK', 'BMY', 'NVS', 'PFE'])]
#within one earnings call
# graph_df=emb_transformed[emb_transformed['transcriptid']==40]
#stratify sample on the label 
graph_df=graph_df.groupby(['label','symbol']).apply(lambda x: x.sample(frac=0.1))

# %%
# graph_df=emb_transformed[emb_transformed['label']==0].sample(500)
# graph_df=emb_transformed.iloc[2000:2500]
# graph_df=emb_transformed[emb_transformed['label']==1]
graph_df["symbol"] = pd.Categorical(graph_df["symbol"])
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


