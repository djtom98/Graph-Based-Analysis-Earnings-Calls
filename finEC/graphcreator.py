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
import pickle
import datapreproc as dpp
import finEC.datapreproc as dpp
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import warnings
warnings.filterwarnings('ignore')
import regex as re

# %%
#get the data
ec=pickle.load(open("../data/Sentiment_Market_Pharma/earnings_call_top10_ph.pickle", "rb"))
ec=ec.reset_index()

# %%
cleanedec=dpp.process_ec(ec,0)

# %%
re.search(r'(?i)()')

# %%
# re.findall(,cleanedec.cleanedec[0])
# "(?i)(?:[. ]{1,2})([ a-zA-Z]*)(?::)"gm
pattern=r'(?i)(?:[. ]{0,2})((\b\w+\b[\s\r\n]*){1,3})(?::)'
# Then use the following to get all overlapping indices:
input=cleanedec.cleanedec[0]
indicesTuple = [(mObj.start(1),mObj.end(1)) for mObj in re.finditer(pattern,input)]


# %%
class Transcript:
    def __init__(self,ec):
        self.speakers=[]
        self.text=


# %%
input[indicesTuple[1][0]:indicesTuple[1][1]]

# %%
cleanedec.cleanedec[0].split(':')

# %%
