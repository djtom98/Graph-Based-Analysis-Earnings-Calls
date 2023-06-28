# Make Earnings Calls Great Again: using text-data to predict the stock price of big pharmaceutical companies

This is a work in progress for the master's thesis at the Barcelona School of Economics.

**Group:**

- Davis Thomas
- Lucas Santos
- Vicente Lisboa


**Goal:** predict stock price movement of selected publically trading pharma companies using NLP techniques and tools learnt during the coursework of the masters.

This thesis is being done in collaboration with Novartis, who have been kind enough to present us with this challenge, and who have taken on a guiding role throughout this project.



## Setup Instructions

After downloading the repository, in the root directory, run the following: \
pip install -e . \
Great! You should now be able to run the notebooks. Please raise an issue with your problem and console error warnings if you face problems.


READ.ME OF OTHER GROUP
### Structure repository

- In the folder `Utils` you can find all the functions used for the forecast-classification stage
- In the folder `Notebooks` you can find the different notebooks used in the pre-processing, exploratory data analysis and forecasting stages of the research.

   1. `0_ACLED_preprocessing`: ACLED cleaning process and subsetting for Latin American countries.
   2. `1_twitter_scraping_sentiment_sample`: Process for scraping tweets using Twarc2 and sentiment analysis with Pysentimiento.
   3. `2_twitter_embeddings`: Creation of embedding representation from tweets.
   4. `3_twitter_dictionary`: Creation of the dictionary-based index of conflict.
   5. `4_data_aggregation`: Creation of the final database with all the features used in the analysis.
   6. `5_data_EDA`: Exploratory data analysis
   7. `6_forecast_classification`: Generates forecasts for the classification problem of social unrest prediction. Disclaimer: the results are not          perfectly reproducable due to the stochasticity inherent in some models.



## Data 


The final data (*Data Final ACLED_weekly_total.csv*) contains 635 variables, some of them are: 

- `isocode`: ISO identification code per country
- `best_sum`: Total number of death estimations made by ACLED in all kind of violent and demonstration events (`_p` refers to this variable only for protests and riots, `_o` refers to other violent events).
- `best_mean`: average number of death estimations made by ACLED in all kind of violent and demonstrations events.
- `event_count`: Total number of events recorded by ACLED in all kind of violent and demonstrations events.
- `POS_sum_P`: Weekly sum of tweets with positive sentiment based on politicians' tweets. Same variables are calculated for each sentiment, emotion and hate-speech. 
- `POS_sum_R`: Weekly sum of tweets with positive sentiment based on replies to politicians' tweets. Same variables are calculated for each sentiment, emotion and hate-speech. 
- `embd_0` - `embd_511`: Tweets embedding representation.
- `dict_share` : conflict index based on a dictionary method.
- `index_google`: Google Trends index based on Protests.
