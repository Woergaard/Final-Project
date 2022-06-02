import matplotlib.pyplot as plt
import pandas as pd
import re
import string 
import pickle
import numpy as np
import os 
import ast
import itertools

#!pip install nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords # For Removing Stop words like < the , an , is ,..etc >
n_words= stopwords.words('english') #specify english stop words only
from nltk.tokenize import word_tokenize # for Tokenizing the sentnces as tokens
from nltk.stem.porter import PorterStemmer # converting words to their root forms ,speed and simplicity
porter = PorterStemmer() #Create stemmer obejct
from nltk.stem import WordNetLemmatizer # also converting words to their actual root forms(noun , verb ,aobjective) ,but it slow
lemmatizer = WordNetLemmatizer() #Create lemmatizer obejct
from nltk import ngrams

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.compose import ColumnTransformer