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
- `avg_euclidean_distance`: Euclidean distance between embeddings of the speakers and analysts in a transcript.
- `cosine_similarity` : Between vectors of the topic allocations.
