#%%
#importing files
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
#%%
ec10=pickle.load(open("../data/Sentiment_Market_Pharma/earnings_call_top10_ph.pickle", "rb"))
ec10=ec10.reset_index()
ec10['date']=pd.to_datetime(ec10.date)
mt10=pickle.load(open("../data/Sentiment_Market_Pharma/main_top10_ph.pickle", "rb"))
mt10['date']=pd.to_datetime(mt10.date)
# %%
df=ec10[['content','symbol','date']]
# %%
#getting the stock price a day before the close price
mt10

# %%
# # ec10[(ec10['date']>np.datetime64('2017-01-01')) & (ec10['date']<np.datetime64('2022-12-12'))]
# #just 203 rows in the time period
# date_range = pd.date_range(start=mt10['date'].min(), end=df['date'].max())

# # Create a new dataframe with the date range and merge it with the original dataframe
# new_df = pd.DataFrame({'date': date_range})
# mt10 = pd.merge(new_df, mt10, on='date', how='left')
# %%
merged_df = pd.merge(left=mt10, right=ec10,how='left',left_on=['ticker','date'], right_on=['symbol','date'])

# %%
abbv=merged_df[merged_df['ticker']=='ABBV']
# %%
abbv
abbv['prevclose']=abbv.close.shift(1)  
abbv['3dayclose']=abbv.close.shift(-3)  
abbv['5dayclose']=abbv.close.shift(-5)  
# %%
abbv['3diff']=abbv['3dayclose']-abbv['prevclose']
abbv['5diff']=-abbv['5dayclose']-abbv['prevclose']
abbv=abbv[abbv['symbol'].notna()]
# %%
#getting finbert sentiment analysis
from transformers import BertTokenizer, BertForSequenceClassification,AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# sent_val = list()
# for x in abbv.content.to_list():
inputs = tokenizer(abbv.content.to_list(), return_tensors="pt", padding=True,truncation=True)
outputs = model(**inputs)

import torch


predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

# print(x, '----', val)
print('#######################################################')    
# sent_val.append(val)
# %%
