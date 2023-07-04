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
from sklearn.linear_model import LinearRegression

# %%
df=pd.read_csv("../data/stock_prices.csv",parse_dates=["Date"])

# %%
ec=pickle.load(open("../data/Sentiment_Market_Pharma/earnings_call_top10_ph.pickle", "rb"))
ec=ec.reset_index()
events=ec[['symbol','date','quarter','year']]
events['date']=pd.to_datetime(events.date)
events.sort_values(['symbol','date'],inplace=True)

# %%
sp500=pd.read_csv("../data/SP500HistoricalData.csv",parse_dates=['Date'],thousands=',', decimal='.')
sp500latest=pd.read_csv("../data/HistoricalData_1686903580615.csv",parse_dates=['Date'],thousands=',', decimal='.')
sp500=pd.merge(sp500,sp500latest,how="outer",on="Date")

# %%
sp500=sp500[['Date','Price','Close/Last']]
sp500['Price'].fillna(sp500['Close/Last'], inplace=True)
sp500.drop(columns=['Close/Last'],inplace=True)
#rename the columns
sp500.columns=['Date','SP500']
sp500['Date']=pd.to_datetime(sp500.Date)
sp500.sort_values('Date',inplace=True)
sp500['pct_change_sp500'] = sp500['SP500'].pct_change()


# %%
#merge the two dataframes
df1=df.merge(sp500,how="left",on="Date")

# %%
df1.info()

# %%
df1.Date.min(),df1.Date.max()

# %%
df1 = df1.sort_values(['Ticker', 'Date'])

# Apply percentage change to each panel
df1['pct_change_close'] = df1.groupby('Ticker')['Close'].pct_change()

# %%
#let us plot the data for each stock
import matplotlib.pyplot as plt
plt.plot(df1[df1.Ticker=="NVS"].Date,df1[df1.Ticker=="NVS"].pct_change_close)

# %%
plt.plot(df1[df1.Ticker=="NVS"].Date,df1[df1.Ticker=="NVS"].pct_change_sp500)

# %%
#implementing the event study methodology
#first, we need to create the event window
#we will use the 2 

# %%
pd.options.mode.chained_assignment = None  # default='warn'

# %%
for ticker in df1.Ticker.unique():
    df=df1[df1.Ticker==ticker]
    event_dates=events[events.symbol==ticker].date.to_list()
        
    # Define the event window
    event_window = 2

    # Initialize a list to store the abnormal returns
    abnormal_returns_pre = []
    abnormal_returns_event = []
    abnormal_returns_post = []
    allCAR1=[]
    allCAR2=[]
    allMAR1=[]
    allMAR2=[]
    pre_event_window = 100
    post_event_window = 30
    # Iterate over the event dates
    for event_date in event_dates:
        # Filter the data within the event window
        event_start = event_date - pd.Timedelta(days=event_window)
        event_end = event_date + pd.Timedelta(days=event_window)
        
        event_data = df[(df['Date'] >= event_start) & (df['Date'] <= event_end)]

        # Filter the data before the event
        pre_event_start = event_start - pd.Timedelta(days=pre_event_window)
        pre_event_data = df[(df['Date'] >= pre_event_start) & (df['Date'] < event_start)]
        #dropna
        pre_event_data.dropna(inplace=True)

        #pead
        post_event_end = event_end + pd.Timedelta(days=post_event_window)
        post_event_data = df[(df['Date'] > event_end) & (df['Date'] <= post_event_end)]
        # Fit the linear regression model using the market model
        X = pre_event_data['pct_change_sp500'].values.reshape(-1, 1)  # Replace 'Market Return' with your market return column
        y = pre_event_data['pct_change_close'].values.reshape(-1, 1)  # Replace 'Stock Return' with your stock return column
        model = LinearRegression()
        model.fit(X, y)

        # Predict stock returns for each day in the event window
        X_event = event_data['pct_change_sp500'].values.reshape(-1, 1)
        predicted_returns = model.predict(X_event)

        # Compute abnormal returns for each day in the event window
        actual_returns = event_data['pct_change_close'].values
        abnormal_returns_event.extend(actual_returns - predicted_returns.flatten())
        #compute cumulative abnormal returns
        CAR1 = np.sum(abnormal_returns_event)
        allCAR1.append(CAR1)
        #compute mean abnormal returns
        MAR1 = np.mean(abnormal_returns_event)
        allMAR1.append(MAR1)

        #compute the abnormal returns for the pre event window
        X_pre = pre_event_data['pct_change_sp500'].values.reshape(-1, 1)
        predicted_returns_pre = model.predict(X_pre)
        actual_returns_pre = pre_event_data['pct_change_close'].values
        abnormal_returns_pre.extend(actual_returns_pre - predicted_returns_pre.flatten())

        #compute the abnormal returns for the post event window
        X_post = post_event_data['pct_change_sp500'].values.reshape(-1, 1)
        predicted_returns_post = model.predict(X_post)
        actual_returns_post = post_event_data['pct_change_close'].values
        abnormal_returns_post.extend(actual_returns_post - predicted_returns_post.flatten())
        #compute cumulative abnormal returns
        CAR2 = np.sum(abnormal_returns_post)
        allCAR2.append(CAR2)
        #compute mean abnormal returns
        MAR2 = np.mean(abnormal_returns_post)
        allMAR2.append(MAR2)

    events.loc[events.symbol==ticker,'CAR1']=allCAR1
    events.loc[events.symbol==ticker,'CAR2']=allCAR2
    events.loc[events.symbol==ticker,'MAR1']=allMAR1
    events.loc[events.symbol==ticker,'MAR2']=allMAR2
    # join the abnormal returns with the daily stock returns
    # df1[df1.Ticker==ticker][(df1['Date'] >= pre_event_start) & (df1['Date'] <= post_event_end)]['abnormalreturn']=abnormal_returns_pre + abnormal_returns_event+abnormal_returns_post



# %%
# importing the data

df = pd.read_csv('../data/stock_prices_ec.csv',parse_dates=['Date'])

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# creating the relative for the EPS

df['eps_surprise'] = (df['eps_mean_report'] / df['eps_mean']) - 1
df=df.merge(events, how='left', left_on=['Ticker', 'Date'], right_on=['symbol', 'date'])

# %%
#adding one hot encoding for the years 2008 and 2020
df['year_2008'] = np.where(df['year'] == 2008, 1, 0)
df['year_2020'] = np.where(df['year'] == 2020, 1, 0)
#adding one hot encoding for the quarters
df['quarter_1'] = np.where(df['quarter'] == 1, 1, 0)
df['quarter_2'] = np.where(df['quarter'] == 2, 1, 0)
df['quarter_3'] = np.where(df['quarter'] == 3, 1, 0)
df['quarter_4'] = np.where(df['quarter'] == 4, 1, 0)
#adding one hot encoding for the different tickers using the get_dummies function
df=pd.get_dummies(df, columns=['Ticker'], prefix=['Ticker'])

# %%
# defining features
#targets are CAR1, CAR2 cumulative abnormal returns and MAR1,MAR2 mean abnormal returns
#CAR2 is the best target. CAR1 is the cumulative abnormal returns for the event window and CAR2 is the cumulative abnormal returns for the post event window
target = ['CAR2']

identification = ['Date', 'Ticker']
dummies=['Ticker_ABBV', 'Ticker_AZN',
       'Ticker_BMY', 'Ticker_JNJ', 'Ticker_LLY', 'Ticker_MRK', 'Ticker_NVO',
       'Ticker_NVS', 'Ticker_PFE', 'Ticker_ROG']
# dummies=['year_2008','year_2020','quarter_1','quarter_2','quarter_3','quarter_4','Ticker_ABBV', 'Ticker_AZN',
#        'Ticker_BMY', 'Ticker_JNJ', 'Ticker_LLY', 'Ticker_MRK', 'Ticker_NVO',
#        'Ticker_NVS', 'Ticker_PFE', 'Ticker_ROG']
surprise = ['eps_surprise']

technical_features = ['rsi5', 'rsi14', 'rsi50', 'rsi200', 'por_change_1_week_val', 'por_change_1_month_val', 'por_change_1_quater_val', 'por_change_1_year_val']

topic_features = ['t1', 't2', 't3', 't4', 't5', 't6' ] # how much of the topic is there on the document
topic_features_d = ['t1d', 't2d', 't3d', 't4d', 't5d', 't6d'] # how much of the topic is there on the document and in which direction (if it's positive or negative)
topic_features_dg = ['t1dg', 't2dg', 't3dg', 't4dg', 't5dg', 't6dg'] # topic_features_d + if they are talking in the short term or long term

# %%
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  GridSearchCV, HalvingGridSearchCV, train_test_split

# %% [markdown]
# ### Linear Model

# %%
# train test split
df = df.dropna()


X = df[surprise+dummies] # model with technical features and surprise
# X = df[surprise + technical_features + topic_features] # model with technical features and surprise
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2000) #splitting randomly, it's correct?

# %%
# Add a constant to the features
X_train1 = sm.add_constant(X_train)
X_test1 = sm.add_constant(X_test)

# Create the linear model and complete the least squares fit
model = sm.OLS(y_train, X_train1)
results = model.fit()  # fit the model
print(results.summary())

print(results.pvalues)


# Make predictions from our model for train and test sets
train_predictions = results.predict(X_train1)
test_predictions = results.predict(X_test1)

# %%
# Scatter the predictions vs the targets with 20% opacity
plt.scatter(train_predictions, y_train, alpha=0.2, color='b', label='train')
plt.scatter(test_predictions, y_test, alpha=0.2, color='r', label='test')

# Plot the perfect prediction line
xmin, xmax = plt.xlim()
plt.plot(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01), c='k')

# Set the axis labels and show the plot
plt.xlabel('predictions')
plt.ylabel('actual')
plt.legend()  # show the legend
plt.show()

# %% [markdown]
# ### Machine Learning methods

# %%
SEED = 2000
# models
models = [
          DecisionTreeRegressor(),
          RandomForestRegressor(),
          AdaBoostRegressor(),
         GradientBoostingRegressor()
         ]

# models parameters

dectree_params = {'max_features': range(4, 10),
                  'max_depth': range(3, 6),
                  'min_samples_leaf': range(5, 10),
                  'random_state': [SEED]}

randomforest_params = {'max_features': range(4, 10),
                  'n_estimators': [50, 100],
                  'max_depth': range(3, 6),
                  'min_samples_leaf': range(5, 10),
                  'random_state': [SEED]}

adab_params = {'learning_rate': [0.05, 1],
                  'n_estimators': [50, 100],
                  'loss':['linear', 'square'],
                  'random_state': [SEED]}

gb_params = {'learning_rate': [0.05, 1],
                  'n_estimators': [50, 100],
                  'max_depth': range(3, 6),
                'max_features': range(4, 10),
                  'random_state': [SEED]}

params = [ dectree_params, randomforest_params, adab_params, gb_params]
names = ['DecisionTree_Regressor', 'RandomForest_Regressor', 'AdaBoost_Regressor',
        'GradientBoosting_Regressor']
scores = {}

# gridsearch
for i, model in enumerate(models):
    print(f"Grid-Searching for model {names[i]}...")
    best_model = HalvingGridSearchCV(model, params[i], n_jobs=4, cv=5, 
                              scoring='neg_root_mean_squared_error', verbose = 0) #using RMSE as metric for scoring
    best_model.fit(X_train, y_train)
    print(f"Best model fitted")
    #assign the best parameters to my models
    models[i].set_params(**best_model.best_params_)

    print(f'{names[i]} chosen hyperparameters: {best_model.best_params_}')
    print(f'{names[i]} RMSE score on train sample: {-best_model.best_score_}')

# %%
# getting the predictions for the best model
train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

plt.scatter(train_predictions, y_train, alpha=0.2, color='b', label='train')
plt.scatter(test_predictions, y_test, alpha=0.2, color='r', label='test')

# Plot the perfect prediction line
xmin, xmax = plt.xlim()
plt.plot(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01), c='k')

plt.xlabel('predictions')
plt.ylabel('actual')
plt.legend()  
plt.show()
