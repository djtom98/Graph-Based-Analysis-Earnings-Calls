# Decoding Abnormal Returns: Unraveling Insights from Pharmaceutical Sector Earnings Calls through Graph-Enhanced Text Analysis

This repository contains the supporting code for our master's thesis at the Barcelona School of Economics.
2022-2023

**Group:**

- Davis Thomas
- Lucas Santos
- Vicente Lisboa


**Goal:** predict stock price movement of selected publically trading pharma companies using NLP techniques and tools learned during the coursework of the masters.

This thesis is being done in collaboration with Novartis, who have been kind enough to present us with this challenge, and who have taken on a guiding role throughout this project.



## Setup Instructions

After downloading the repository, in the root directory, run the following: \
pip install -e . \
Great! You should now be able to run the notebooks. Please raise an issue with your problem and console error warnings if you face problems.



### Structure of repository

- In the folder `finEC` you can find all the functions used for data preprocessing and graph creation.
- In the folder `Notebooks` you can find the different notebooks used in the pre-processing, exploratory data analysis, and prediction stages of the research.

   1. `5- error_analysis.ipynb`: Merging datasets to grid-search over several models to determine the best-performing ML model.
   2. `10-models and results.ipynb`: Contains the final models, comparison of models over different metrics, and analysis of the results.
   3. `8-LDAcosine.ipynb`: Cosine similarities using LDA vectors and TF-IDF vectors are created in this notebook.
   4. `9-topicmodeling.ipynb`: Dictionary based topic modeling features are generated using this notebook.
   



## Data 


The final data is present in the /data folder. 
- `graph` folder contains the data generated after processing the text to transform it into a graph representation.

  
