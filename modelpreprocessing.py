# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:40:14 2021

@author: EricH
"""
#column names are:
# Unnamed: 0	review_body	review_date	hotelName	hotelUrl

import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
import os
nlp = spacy.load("en_core_web_sm")
from spacy import displacy

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

stopwords=list(STOP_WORDS)
from string import punctuation
punctuation=punctuation+ '\n'

import pandas as pd
import scipy.spatial
import pickle as pkl
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


embedder = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv('C:/Users/EricH/MachineLearning/Assignment3/sydneyhotels.csv')

df['hotelName'].value_counts()
df['hotelName'].drop_duplicates()

df_combined = df.sort_values(['hotelName']).groupby('hotelName', sort=False).review_body.apply(''.join).reset_index(name='all_review')

# re combines and puts everything in lower case
import re
from tqdm import tqdm

df_combined['all_review'] = df_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

df_combined['all_review']= df_combined['all_review'].apply(lambda x: lower_case(x))

df = df_combined

df_sentences = df_combined.set_index("all_review")

df_sentences = df_sentences["hotelName"].to_dict()
df_sentences_list = list(df_sentences.keys())
len(df_sentences_list)

list(df_sentences.keys())[:5]

df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]


# gives an encoding of the full concatenated list of corpus
corpus = df_sentences_list
corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)

paraphrases = util.paraphrase_mining(model, corpus)


from sentence_transformers import SentenceTransformer, util
import torch

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)


with open("corpus.pkl" , "wb") as file1:
  pkl.dump(corpus,file1)
with open("corpus_embeddings.pkl" , "wb") as file2:
  pkl.dump(corpus_embeddings,file2)
with open("df.pkl" , "wb") as file3:
  pkl.dump(df,file3)


#query_embeddings_p =  util.paraphrase_mining(model, queries,show_progress_bar=True)

# import pickle as pkl
#upload a csv file, convereted that csv file after cleaning and converted to embedding
# if we dump the corpus into a pickle file, and then load the file and it will be saved in the system. 
# we havec saved the model, so we dont ahve to ever run the corpus 
# pkl.load()
# with open("/content/drive/MyDrive/BertSentenceSimilarity/Pickles/corpus_embeddings.pkl" , "wb") as file_:
#  pkl.dump(corpus_embeddings,file_)
