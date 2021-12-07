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

df = pd.read_csv('C:/Users/EricH/MachineLearning/Assignment3/Sydneyhotelreviews.csv')

df['hotelName'].value_counts()
a = df['hotelName'].drop_duplicates()

new_values = ['Novotel Sydney on Darling Harbour',
'Meriton Suites Zetland',
'Cambridge Hotel Sydney',
'Vibe Hotel Rushcutters Bay Sydney',
'Wake Up! Sydney',
'Vulcan Hotel Sydney',
'Mercure Sydney',
'Metro Aspire Hotel Sydney',
'Novotel Sydney Central',
'Shangri-La Hotel Sydney',
'Hotel Challis',
'Novotel Sydney Darling Square',
'Adina Apartment Hotel Coogee Sydney',
'Sydney Harbour YHA',
'The Russell Boutique Hotel',
'Bondi Beach House Accommodation',
'28 Hotel',
'Four Points by Sheraton, Sydney Central Park',
'Central Railway Hotel',
'The Ultimo',
'West Hotel Sydney, Curio Collection by Hilton',
'Meriton Suites Waterloo',
'Adina Apartment Hotel Sydney Central',
'Coogee Sands Hotel & Apartments',
'Macleay Hotel',
'Veriu Central',
'Little Albion, a Crystalbrook Collection Bouti…',
'The Maisonette',
'Hyde Park Inn',
'Veriu Broadway']

old_values = ["""0    Novotel Sydney on Darling Harbour 
              Name: hotel_name, dtype: object""",
              """1    Meriton Suites Zetland
Name: hotel_name, dtype: object""",
"""2    Cambridge Hotel Sydney
Name: hotel_name, dtype: object""",
"""3    Vibe Hotel Rushcutters Bay Sydney
Name: hotel_name, dtype: object""",
"""4    Wake Up! Sydney
Name: hotel_name, dtype: object""",
"""5    Vulcan Hotel Sydney
Name: hotel_name, dtype: object""",
"""6    Mercure Sydney
Name: hotel_name, dtype: object""",
"""7    Metro Aspire Hotel Sydney
Name: hotel_name, dtype: object""",
"""8    Novotel Sydney Central
Name: hotel_name, dtype: object""",
"""9    Shangri-La Hotel Sydney
Name: hotel_name, dtype: object""",
"""10    Hotel Challis
Name: hotel_name, dtype: object""",
"""11    Novotel Sydney Darling Square
Name: hotel_name, dtype: object""",
"""12    Adina Apartment Hotel Coogee Sydney
Name: hotel_name, dtype: object""",
"""13    Sydney Harbour YHA
Name: hotel_name, dtype: object""",
"""14    The Russell Boutique Hotel
Name: hotel_name, dtype: object""",
"""15    Bondi Beach House Accommodation
Name: hotel_name, dtype: object""",
"""16    28 Hotel
Name: hotel_name, dtype: object""",
"""17    Four Points by Sheraton, Sydney Central Park
Name: hotel_name, dtype: object""",
"""18    Central Railway Hotel
Name: hotel_name, dtype: object""",
"""19    The Ultimo
Name: hotel_name, dtype: object""",
"""20    West Hotel Sydney, Curio Collection by Hilton
Name: hotel_name, dtype: object""",
""""21    Meriton Suites Waterloo
Name: hotel_name, dtype: object""",
"""22    Adina Apartment Hotel Sydney Central
Name: hotel_name, dtype: object""",
"""23    Coogee Sands Hotel & Apartments
Name: hotel_name, dtype: object""",
"""24    Macleay Hotel
Name: hotel_name, dtype: object""",
"""25    Veriu Central
Name: hotel_name, dtype: object""",
"""26    Little Albion, a Crystalbrook Collection Bouti...
Name: hotel_name, dtype: object""",
"""27    The Maisonette
Name: hotel_name, dtype: object""",
"""28    Hyde Park Inn
Name: hotel_name, dtype: object""",
"""29    Veriu Broadway
Name: hotel_name, dtype: object"""]

df['hotelName'] = df['hotelName'].replace(old_values,new_values)



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
