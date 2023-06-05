# %%
import pandas as pd
import pickle
# import numpy as np # linear algebra
# # import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline
from datetime import datetime
import regex as re
import pickle

# %%
ec10=pickle.load(open("../data/Sentiment_Market_Pharma/earnings_call_top10_ph.pickle", "rb"))
ec10=ec10.reset_index()

# %%
def preprocess_ec(x):
    x.replace('Operator:','')
    return x


# %%
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
import gensim
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# %%
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# %%
txt=ec10.content.tolist()

# %%
# simplify Penn tags to n (NOUN), v (VERB), a (ADJECTIVE) or r (ADVERB)
def simplify(penn_tag):
    pre = penn_tag[0]
    if (pre == 'J'):
        return 'a'
    elif (pre == 'R'):
        return 'r'
    elif (pre == 'V'):
        return 'v'
    else:
        return 'n'

# %%
toks = gensim.utils.simple_preprocess(str(txt), deacc=True)
# toks


# %%
nltk.download('stopwords')
nltk.download('omw-1.4')

def preprocess(text):
    stop_words = stopwords.words('english')
    toks = gensim.utils.simple_preprocess(str(text), deacc=True)
    wn = WordNetLemmatizer()
    # return [wn.lemmatize(tok, simplify(pos)) for tok, pos in nltk.pos_tag(toks) if tok not in stop_words]
    return [tok for tok in toks if tok not in stop_words]

# %%
corp = [preprocess(line) for line in txt]
# corp

# %%
dictionary = gensim.corpora.Dictionary(corp)
len(dictionary)

# %%
def viz_model(model, modeldict):
    ntopics = model.num_topics
    # top words associated with the resulting topics
    topics = ['Topic : '.format(t,modeldict[w]) 
              for t in range(ntopics) 
              for w,p in model.get_topic_terms(t, topn=1)]
              
    terms = [modeldict[w] for w in modeldict.keys()]
    fig,ax=plt.subplots()
    ax.imshow(model.get_topics())  # plot the numpy matrix
    ax.set_xticks(modeldict.keys())  # set up the x-axis
    ax.set_xticklabels(terms, rotation=90)
    ax.set_yticks(np.arange(ntopics))  # set up the y-axis
    ax.set_yticklabels(topics)
    plt.show()

# %%
def test_eta(eta, dictionary, ntopics, print_topics=True, print_dist=True):
    np.random.seed(42) # set the random seed for repeatability
    bow = [dictionary.doc2bow(line) for line in corp] # get the bow-format lines with the set dictionary
    with (np.errstate(divide='ignore')):  # ignore divide-by-zero warnings
        model = gensim.models.ldamodel.LdaModel(
            corpus=bow, id2word=dictionary, num_topics=ntopics,
            random_state=42, chunksize=100, eta=eta,
            eval_every=-1, update_every=1,
            passes=150, alpha='auto', per_word_topics=True)
    # visualize the model term topics
    viz_model(model, dictionary)
    print('Perplexity: :.2f'.format(model.log_perplexity(bow)))
    if print_topics:
        # display the top terms for each topic
        for topic in range(ntopics):
            print('Topic : '.format(topic, [dictionary[w] for w,p in model.get_topic_terms(topic, topn=3)]))
    if print_dist:
        # display the topic probabilities for each document
        for line,bag in zip(txt,bow):
            doc_topics = ['(, :.1%)'.format(topic, prob) for topic,prob in model.get_document_topics(bag)]
            print(' '.format(line, doc_topics))
    return model

# %%
#this cell creates a very low prior for all terms not in the dictionary (priors) and a very high prior for terms found
#in the dictionary
#watch out that you need to pass a dictionary that is pre-processed the same way you are pre-processing the text

def create_eta(priors, etadict, ntopics):
    eta = np.full(shape=(ntopics, len(etadict)), fill_value=0.0001) # create a (ntopics, nterms) matrix and fill with low number
    for word, topic in priors.items(): # for each word in the list of priors
        keyindex = [index for index,term in etadict.items() if term==word] # look up the word in the dictionary
        if (len(keyindex)>0): # if it's in the dictionary
            eta[topic,keyindex[0]] = 500  # put a large number in there
    return eta

# %%
# test_eta('auto', dictionary, ntopics=6)

# %%
#emulating the sp500 quant report on ideal topics and bidirectional tags to extract from earnings calls
t1_revenue=['sales', 'revenue', 'top line', 'top bottom line', 'net revenue', 'organic revenue growth', 'organic sales growth', 'operational sales']
t2_earnings=['eps', 'earnings', 'earnings per share',
'net income', 'bottom line', 'top bottom line']
t3_profitability=['margin', 'gross margin', 'operating margin', 'return invested capital', 'return capital']
t4_operatingincome=['ebit', 'operating income', 'operating profit', 'operating earning']
t5_cashflow=['cash flow', 'operating cash flow', 'cash flow operations',
'free cash flow']
t6_shareholderreturn=['buyback', 'dividends', 'dividend per share', 'share repurchase', 'repurchased million shares']

# %%
uniontopics=t1_revenue+t2_earnings+t3_profitability+t4_operatingincome+t5_cashflow+t6_shareholderreturn

# %%
for word in uniontopics[0:3]:
    mask=ec10.content.str.contains(word)
    result=ec10[mask].content
    print(f"{word} exists in the following rows:\n{result.count()}\n")

# %% [markdown]
# Okay so a casual investigation of the data reveals that there are sufficient mentions of these terms within the earnings calls to justify a topic modeling based on this approach. Let's give this a try!

# %%
#struggling with getting trigrams into topic modeling topics
#should I get the lists into the 
alltopic_format=[]
for word in uniontopics:
    alltopic_format.append('_'.join(word.split(' ')))

# %%
t1_revenue_f=[]
t2_earnings_f=[]
t3_profitability_f=[]
t4_operatingincome_f=[]
t5_cashflow_f=[]
t6_shareholderreturn_f=[]


# %%
for word in t1_revenue:
    t1_revenue_f.append('_'.join(word.split(' ')))
for word in t2_earnings:
    t2_earnings_f.append('_'.join(word.split(' ')))
for word in t3_profitability:
    t3_profitability_f.append('_'.join(word.split(' ')))
for word in t4_operatingincome:
    t4_operatingincome_f.append('_'.join(word.split(' ')))
for word in t5_cashflow:
    t5_cashflow_f.append('_'.join(word.split(' ')))
for word in t6_shareholderreturn:
    t6_shareholderreturn_f.append('_'.join(word.split(' ')))

# %%
trigram_bigram_root=[]
for word in ['_'.join(x.split('_')[0:2]) for x in alltopic_format if x.count('_') > 1 ]:
    trigram_bigram_root.append(word)
    

# %%
bigrams_pure=[]
for word in ['_'.join(x.split('_')[0:2]) for x in alltopic_format if x.count('_') == 1 ]:
    bigrams_pure.append(word)
    

# %%
unigrams_pure=[]
for word in [x for x in alltopic_format if x.count('_') <1 ]:
    unigrams_pure.append(word)
unigrams_pure=list(set(unigrams_pure))

# %%
bigram = gensim.models.Phrases(corp, min_count=1, threshold=5) # higher threshold fewer phrases.

# %%
frozen_bigram = bigram.freeze()  # freeze bigrams' scores for compactness/efficiency
for bigramword in trigram_bigram_root:
    frozen_bigram.phrasegrams[bigramword] = float('inf')
for unigramword in unigrams_pure:
    frozen_bigram.phrasegrams[unigramword] = float('inf')
for bigramword in bigrams_pure:
    frozen_bigram.phrasegrams[bigramword] = float('inf')

# %%
trigram = gensim.models.Phrases(frozen_bigram[corp], threshold=1) 
trifreeze=trigram.freeze()
for word in alltopic_format:
    trifreeze.phrasegrams[word]=float('Inf')


# trigram_mod = gensim.models.phrases.Phraser(trifreeze)
words_trigram = [trifreeze[doc] for doc in corp]
dictionarytrigram = gensim.corpora.Dictionary(words_trigram)
len(dictionarytrigram)
#55949

# %%
apriori_original = dict()
for a in t1_revenue_f:
    apriori_original[a]=0

for b in t2_earnings_f:

    apriori_original[b]=1

for c in t3_profitability_f:
    
    apriori_original[c]=2

for d in t4_operatingincome_f:

    apriori_original[d]=3
for e in t5_cashflow_f:

    apriori_original[e]=4

for f in t6_shareholderreturn_f:
    apriori_original[f]=5

# for b in 

# t1_revenue_f=[]
# t2_earnings_f=[]
# t3_profitability_f=[]
# t4_operatingincome_f=[]
# t5_cashflow_f=[]
# t6_shareholderreturn_f=[]
#generate the eta vector
# eta = create_eta(apriori_original, dictionary, 10)

# %%
apriori_original

# %%
# trigram = gensim.models.Phrases(frozen_phrases[corp], threshold=10) 

# %%
# trifreeze=trigram.freeze()
# trifreeze.phrasegrams['top_bottom_line']=float('Inf')
# trifreeze.phrasegrams['organic_revenue_growth']=float('Inf')
# trifreeze.phrasegrams['organic_sales_growth']=float('Inf')
# trifreeze.phrasegrams['earnings_per_share']=float('Inf')
# trifreeze.phrasegrams['return_invested_capital']=float('Inf')


# %%
# trigram_mod = gensim.models.phrases.Phraser(trifreeze)
# words_trigram = [trifreeze[doc] for doc in corp]

# %%
eta = create_eta(apriori_original, dictionarytrigram,6)

# %%
# ldamodel_1=test_eta(eta, dictionarytrigram, ntopics=6)

# %%
# ldamodel_1.print_topics(num_topics=6, num_words=15)

# %%
# import pickle

# filehandler = open("../data/ldamodel.pkl","wb")
# pickle.dump(ldamodel_1,filehandler)
# filehandler.close()


# %%

filehandler=open("../data/ldamodel.pkl","rb")
ldamodel_1=pickle.load(filehandler)
filehandler.close()

# %%
topicallocation=[]
for doc in words_trigram:
    topicallocation.append(ldamodel_1.get_document_topics(bow=dictionarytrigram.doc2bow(doc)))

# %%
for i,t in enumerate(topicallocation):
    for topic, prob in t:
        topic='topic'+str(topic)
        ec10.loc[ec10.index[i],topic]=prob

# %%
ec10=ec10.fillna(0)

# %%
ldamodel_1.get_document_topics(bow=dictionarytrigram.doc2bow(words_trigram[10]))

# %%
# test_eta('auto', dictionarytrigram, ntopics=6)

# %%
#dictionary based topic modeling
positive_words=[ 'increase', 'increased', 'increases', 'increasing', 'increasingly', 'expand', 'expanded', 'expanding', 'expands', 'expansion', 'expansions', 'grow', 'grows', 'grew', 'growth', 'growths', 'improve', 'improved', 'improves', 'improvement', 'improvements', 'strong', 'stronger', 'strongest', 'strongly' ]
negative_words=['decline', 'declined', 'declines', 'declining', 'deteriorate', 'deteriorates', 'deteriorated', 'deteriorating', 'compress', 'compressed', 'compresses', 'compressing', 'compressible', 'compression', 'reduce', 'reduces', 'reduced', 'reducing', 'reduction', 'reductions', 'weak', 'weaker', 'weakest', 'weaken', 'weakens', 'weakened', 'weakening', 'weakness', 'weaknesses']
guidance_words=['full year outlook', 'full year expect', 'guidance', 'outlook', 'forecast', 'expect', 'expects', 'expected', 'expecting', 'expectation', 'expectations']

# %%
t1_revenue_f
t2_earnings_f
t3_profitability_f
t4_operatingincome_f
t5_cashflow_f
t6_shareholderreturn_f


# %%
ec10.date.max()

# %%
#cosine similarity based on the topic modeling
from sklearn.metrics.pairwise import cosine_similarity
ec10

# %%
t1_revenue

# %%
nltk.sent_tokenize(ec10['content'].iloc[0])

# %%
def topiccount(doc):
    c1,c2,c3,c4,c5,c6,i=0,0,0,0,0,0,0

    for sent in nltk.sent_tokenize(doc):
        if any(word in sent for word in t1_revenue):
            c1+=1
        if any(word in sent for word in t2_earnings):
            c2+=1
        if any(word in sent for word in t3_profitability):
            c3+=1
        if any(word in sent for word in t4_operatingincome):
            c4+=1
        if any(word in sent for word in t5_cashflow):
            c5+=1
        if any(word in sent for word in t6_shareholderreturn):
            c6+=1
        i+=1
    return c1/i,c2/i,c3/i,c4/i,c5/i,c6/i


# %%
def directionalitycount(doc):
    c1,c2,c3,c4,c5,c6,i=0,0,0,0,0,0,0

    for sent in nltk.sent_tokenize(doc):
        if any(word in sent for word in t1_revenue) and any(word in sent for word in positive_words) and any(word in sent for word in guidance_words):
            c1+=1
        if any(word in sent for word in t1_revenue) and any(word in sent for word in negative_words) and any(word in sent for word in guidance_words):
            c1-=1
        if any(word in sent for word in t2_earnings) and any(word in sent for word in positive_words) and any(word in sent for word in guidance_words):
            c2+=1
        if any(word in sent for word in t2_earnings) and any(word in sent for word in negative_words) and any(word in sent for word in guidance_words):
            c2-=1
        if any(word in sent for word in t3_profitability) and any(word in sent for word in positive_words) and any(word in sent for word in guidance_words):
            c3+=1
        if any(word in sent for word in t3_profitability) and any(word in sent for word in negative_words) and any(word in sent for word in guidance_words):
            c3-=1
        if any(word in sent for word in t4_operatingincome) and any(word in sent for word in positive_words) and any(word in sent for word in guidance_words):
            c4+=1
        if any(word in sent for word in t4_operatingincome) and any(word in sent for word in negative_words) and any(word in sent for word in guidance_words):
            c4-=1
        if any(word in sent for word in t5_cashflow) and any(word in sent for word in positive_words) and any(word in sent for word in guidance_words):
            c5+=1
        if any(word in sent for word in t5_cashflow) and any(word in sent for word in negative_words) and any(word in sent for word in guidance_words):
            c5-=1
        if any(word in sent for word in t6_shareholderreturn) and any(word in sent for word in positive_words) and any(word in sent for word in guidance_words):
            c6+=1
        if any(word in sent for word in t6_shareholderreturn) and any(word in sent for word in negative_words) and any(word in sent for word in guidance_words):
            c6-=1
        i+=1
    return c1/i,c2/i,c3/i,c4/i,c5/i,c6/i


# %%
def guidancecount(doc):
    c1,c2,c3,c4,c5,c6,i=0,0,0,0,0,0,0

    for sent in nltk.sent_tokenize(doc):
        if any(word in sent for word in t1_revenue) and any(word in sent for word in positive_words):
            c1+=1
        if any(word in sent for word in t1_revenue) and any(word in sent for word in negative_words):
            c1-=1
        if any(word in sent for word in t2_earnings) and any(word in sent for word in positive_words):
            c2+=1
        if any(word in sent for word in t2_earnings) and any(word in sent for word in negative_words):
            c2-=1
        if any(word in sent for word in t3_profitability) and any(word in sent for word in positive_words):
            c3+=1
        if any(word in sent for word in t3_profitability) and any(word in sent for word in negative_words):
            c3-=1
        if any(word in sent for word in t4_operatingincome) and any(word in sent for word in positive_words):
            c4+=1
        if any(word in sent for word in t4_operatingincome) and any(word in sent for word in negative_words):
            c4-=1
        if any(word in sent for word in t5_cashflow) and any(word in sent for word in positive_words):
            c5+=1
        if any(word in sent for word in t5_cashflow) and any(word in sent for word in negative_words):
            c5-=1
        if any(word in sent for word in t6_shareholderreturn) and any(word in sent for word in positive_words):
            c6+=1
        if any(word in sent for word in t6_shareholderreturn) and any(word in sent for word in negative_words):
            c6-=1
        i+=1
    return c1/i,c2/i,c3/i,c4/i,c5/i,c6/i


# %%
topicdictcount=ec10.copy()

# %%
topicdictcount['result']=topicdictcount.content.apply(topiccount)
topicdictcount[['t1','t2','t3','t4','t5','t6']]=topicdictcount['result'].to_list()

# %%
topicdictcount[['t1d','t2d','t3d','t4d','t5d','t6d']]=topicdictcount.content.apply(directionalitycount).to_list()

# %%
topicdictcount[['t1dg','t2dg','t3dg','t4dg','t5dg','t6dg']]=topicdictcount.content.apply(guidancecount).to_list()

# %%
# topicdictcount[['t1d','t2d','t3d','t4d','t5d','t6d']].describe()

# %%



