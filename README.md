# Decoding Abnormal Returns: Unraveling Insights from Pharmaceutical Sector Earnings Calls through Graph-Enhanced Text Analysis

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

- In the folder `finEC` you can find all the functions used for data preprocessing and graph creation.
- In the folder `Notebooks` you can find the different notebooks used in the pre-processing, exploratory data analysis and forecasting stages of the research.

   1. `1 - generate_dataframe.ipynb`: Generating dataframe.
   2. `2 - data_analysis.ipynb`: Data analysis.
   3. `4-djt-CARgen.py`: Creation of target variable.
   4. `5- error_analysis.ipynb`: Error analysis.
   5. `6-topicmodeling.ipynb`: Dictionary based topic modeling.
   6. `models and results.ipynb`: Models
   



## Data 


The final data is present in the \data folder.
Features:
- `avg_euclidean_distance`: ISO identification code per country
- `best_sum`: Total number of death estimations made by ACLED in all kind of violent and demonstration events (`_p` refers to this variable only for protests and riots, `_o` refers to other violent events).
- `best_mean`: average number of death estimations made by ACLED in all kind of violent and demonstrations events.
- `event_count`: Total number of events recorded by ACLED in all kind of violent and demonstrations events.
- `POS_sum_P`: Weekly sum of tweets with positive sentiment based on politicians' tweets. Same variables are calculated for each sentiment, emotion and hate-speech. 
- `POS_sum_R`: Weekly sum of tweets with positive sentiment based on replies to politicians' tweets. Same variables are calculated for each sentiment, emotion and hate-speech. 
- `embd_0` - `embd_511`: Tweets embedding representation.
- `dict_share` : conflict index based on a dictionary method.
- `index_google`: Google Trends index based on Protests.
