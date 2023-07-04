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
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
tokenizercleanup = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")


# %%
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# %%
def cleanup(ec):
    '''remove punctuation and non-ascii characters from the earnings call transcript'''
    # from transformers import AutoTokenizer
    ec=tokenizercleanup.backend_tokenizer.normalizer.normalize_str(ec)
    ec=''.join(s for s in ec if ord(s)>31 and ord(s)<126)
    
    return ec

# %%
def get_mgmt(ec):
    '''Still a work in progress, 21 transcripts are not processed correctly. Aims to get the management presentation from the transcript
    '''
    ecsplit=re.split("operator:", ec, flags=re.IGNORECASE)
    try:
        ec=ecsplit[1]
    except IndexError:
        try:
            ec=re.match(r'(.*)questions(?:[a-zA-Z. ]{0,100}:)',ec).group(0)
        except AttributeError:
            ec=np.nan
    return ec

# %%
def removewords_dict(x,n=1):
    '''removes words from a dictionary of words to remove'''
    # x=x.replace(r'.*(operator[:]{0,1})','')
    x=re.sub(r'(?i)good morning','',x)
    x=re.sub(r'(?i)good afternoon','',x)
    if n==1:
        x=re.sub(r'(?i)operator:','',x)
    # x=x.replace(r'(?i)good morning','',x)
    return 

def get_ner_names(ec):
    '''returns the named entities from the earnings call transcript'''
    tokenizernames = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
    modelnames = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

    nlp = pipeline("ner", model=modelnames, tokenizer=tokenizernames, grouped_entities=True)

    ner_results = nlp(ec)
    names=set()
    for result in ner_results:
         if result['entity_group'] in ['PER'] and result['score']>0.8:
            names.add(result['word'])
    return names

def remove_names(ec):

    ner_results = nlp(ec)
    idx=set()
    for result in ner_results:
        if result['entity'] in ['B-PER','I-PER'] and result['score']>0.5:
            idx=idx.union(set(range(result['start'],result['end'])))
    ec="".join([char for id, char in enumerate(ec) if id not in idx])
    return ec

# %%
def process_ec(ec,level):
    '''Takes in a pandas dataframe and returns a dataframe with cleaned earnings call transcripts
    '''
    if level==0:
        ec['cleanedec']=ec['content'].apply(cleanup)

    elif level==2:
        ec['cleanedec']=ec['content'].apply(cleanup)
        ec['cleanedec']=ec['cleanedec'].apply(remove_names,1)
        ec['cleanedec']=ec['cleanedec'].apply(removewords_dict)
        ec['cleanedec']=ec['cleanedec'].apply(cleanup)
    return ec


def process_ec_v0(ec):
    '''Takes in a pandas dataframe and returns a dataframe with cleaned earnings call transcripts
    '''
    def cleanup(ec):
        '''remove punctuation and non-ascii characters from the earnings call transcript'''
        # from transformers import AutoTokenizer
        ec=tokenizercleanup.backend_tokenizer.normalizer.normalize_str(ec)
        ec=''.join(s for s in ec if ord(s)>31 and ord(s)<126)
        
        return ec

    def get_mgmt(ec):
        '''Still a work in progress, 21 transcripts are not processed correctly. Aims to get the management presentation from the transcript
        '''
        ecsplit=re.split("operator:", ec, flags=re.IGNORECASE)
        try:
            ec=ecsplit[1]
        except IndexError:
            try:
                ec=re.match(r'(.*)questions(?:[a-zA-Z. ]{0,100}:)',ec).group(0)
            except AttributeError:
                ec=np.nan
        return ec

    def removewords_dict(x):
        # x=x.replace(r'.*(operator[:]{0,1})','')
        x=re.sub(r'(?i)good morning','',x)
        x=re.sub(r'(?i)good afternoon','',x)
        x=re.sub(r'(?i)operator:','',x)
        # x=x.replace(r'(?i)good morning','',x)
        return x
    def remove_names(ec):

        ner_results = nlp(ec)
        idx=set()
        for result in ner_results:
            if result['entity'] in ['B-PER','I-PER'] and result['score']>0.8:
                idx=idx.union(set(range(result['start'],result['end'])))
        ec="".join([char for id, char in enumerate(ec) if id not in idx])
        return ec

    ec['cleanedec']=ec['content'].apply(cleanup)
    ec['cleanedec']=ec['cleanedec'].apply(remove_names)
    ec['cleanedec']=ec['cleanedec'].apply(removewords_dict)
    ec['cleanedec']=ec['cleanedec'].apply(cleanup)
    return ec

