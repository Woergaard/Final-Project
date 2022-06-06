# Fake news detection using supervised learning
## Introduction 
This repository contains all the code needed to run the FakeNews predictors which we have developed as part of the Final Project for the course Data Science at the Computer Science Department (DIKU) at University of Copenhagen. 
## Data sources 
* Fake News Corpus: https://github.com/several27/FakeNewsCorpus
* LIAR dataset: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip

## Folder structure 
The folder `databases` contains two .zip file each containing a database for the data scraped of the WikiNews site and data from the FakeNews corpus. 

The folder `fakenews predictor` contains a number of subfolders. The folders are structured in the following way: 

    .
    ├── classification           # Classification files          
    ├── data                     # Data used by the model 
    ├── figs                     # Figures 
    ├── output                   # Output containing information on accuracy, F1-scores, support and recall 
    ├── run.py                   # File running the whole model 
    ├── scripts                  # Scripts containing the code for each model and modules for cleaning data and classifying
    ├── tables                   # LaTeX tables generated as part of the output
    ├── tex                      # LaTeX files summarizing the output in one single PDF 
    └── vectorizer               # Pickled vectorizors for each model 

A number of the folders contains subfolders with the subscript `_content`, `_content_title` and `_title`. If a file or folder contains the subscript `_content` it means that this file or folder contains the data concerning a model trained using only the content data of the FakeNewsCorpus. The subscript `_content_title` means that this file or folder contains the data concerning a model trained using the content and title data of the FakeNewsCorpus and for `_title` that the model is only trained on the title data. 
## Requirements 
In the file `fakenews predictor/requirements.txt` the requirements regarding libraries and packages needed to run the code can be found. 
## Usage
To create the databases make sure that you have unzipped the .zip filed and that you have a user named `postgres` in your postgreSQL installation and that it has the rights to create a database. When this is in order create a database with a name of your choosing. If you are using postgreSQL directly in your terminal, quit your session by `\q` and write the following in your terminal: 

```psql databasename < database```

Where `databasename` is the name of the database you have just created and `database` is the path and name of the database you want to import. Now you have recreated the database with either the FakeNewsCorpus or the WikiNews data. 

To run the fake news predictors we have developed the only thing you have to do is to make sure that the requirements in the `fakenews predictor/requirements.txt` file are met and then run the file `run.py`. When the file has terminated the models have been trained and the file `fakenews predictor/tex/main.pdf` contains an overview of model results across all models. 


## Results 
The accuracy of the models trained on the FakeNewsCorpus can be seen below. Furthermore we have tested the cross-domain performance against the LIAR dataset. As a baseline model we use logistic regression (LR) with a bigram vectorization of the corpus. We further develop a multitude of models making used of TF-IDF, bigram and trigram vectorization combined with classification using $k$-Nearest Neighbours (KNN), Decision Trees (DTC) and Support Vector Machines (SVM) to see if we can be improve the accuracy of prediction compared to the baseline model. The accuracy of the baseline model on the dataset without metadata is 0.871 i.e. 87.1% with an F1-score of 0.843. The best performing model on the dataset without metadata is a model which makes use of TF-IDF vectorization and SVM. The  accuracy of this model is 0.916 i.e. 91.6% with an F1-score of 0.903. These models have been trained and tested on a subset of 75.000 randomly chosen articles of the FakeNewsCorpus. Cross domain performance is tested on the LIAR dataset and shows a decline in performance overall. The accuracy of the baseline model is 0.719 i.e. 71.9% with an F1-score of 0.421. We find that even though 71.9% might seem reasonable it may be misleading as there is a large type II error (false negatives) for all models when predicting i.e. when the true label of the article is FAKE the models predict that the article is REAL. This causes most of the models cross-domain accuracy to lie around 70\% which is close to the actual ratio of REAL/FAKE in the LIAR dataset.

![Table of results](https://github.com/Woergaard/Final-Project/blob/main/fakenews%20predictor/div/final_table.png)
