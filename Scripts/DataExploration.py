import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import re

dataset=pd.read_csv("Redditposts.csv",skiprows=3)
dataset=dataset.drop(columns='Unnamed: 6')

def view_dataset(df):
    return df.head()

def view_shape(df):
    return df.shape

view_dataset(dataset)

