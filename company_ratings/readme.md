## Predicting the Rating of a Company Using Employee Reviews
### Abstract
In this project paper we analyse dataset with user reviews and ratings 
of a few major tech companies. The dataset is using Glassdoor data and 
was available on Kaggle. The goal is to predict a rating (full 1-to-5 star 
or simplified positive/negative) based on user review and other information.
This goal is quite similar to the one of sentiment analysis. So we use traditional
LSTM model with (or without) pretrained embeddings. We use NB as a baseline model.
We get pretty high accuracy (around .8-.9) with some simplified assumptions.

### 1 Introduction 
### 2 Related work
### 3 Dataset
- See detailed description of the dataset in `data_analysis.ipynb`.
- There's a significant problem with this dataset: `summary fiels` is quite 
different from standard reviews (that we use in sentiment analysis). For example:
in many cases it contains the job title, not the review. See detailed analysis in 
`data_summary_cleaning.ipynb`.
- See detailed description of data preprocessing in `data_prep.ipynb`
### 4 RNN model
### 5 Experiments
### 6 Conclusion