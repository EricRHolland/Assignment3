# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:46:41 2021

@author: EricH
"""
import streamlit as st
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
# from collections import Counter
# from heapq import nlargest
# import os
nlp = spacy.load("en_core_web_sm")
# from spacy import displacy

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

stopwords=list(STOP_WORDS) + ['hotel','room','stay','hostel']
punctuation=punctuation+ '\n'

# import scipy.spatial
import pickle as pkl
#import os

from sentence_transformers import SentenceTransformer, util
import torch
st.set_option('deprecation.showPyplotGlobalUse', False)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("Sydney Hotel Search App")
st.markdown("Assignment 3")
st.markdown("This a query search that works using google NLP tools.")
st.markdown("It successfully uses pickle to dump and load data using a preprocessing file.")
st.markdown("This makes the app run much faster after deploying! Woohoo!")

#New preprocessing file will havec the first run, then after that this hotel file will reference the first run
#but wont actually store the data
#instead of how to do it,actually if you do this and do the 2.3.1 if most updated version of spacy
#need to explain what deploying means and how it works with the corpus, was really helpful understanding how it gets encoded
#walk through pkl dump and load and successfull st.markdown call

# this worked the most with the corpus and 
##testing commit

##test

# df = pd.read_csv('C:/Users/EricH/MachineLearning/try2/Assignment3/sydneyhotels.csv')

# df['hotelName'].value_counts()
# df['hotelName'].drop_duplicates()

# df_combined = df.sort_values(['hotelName']).groupby('hotelName', sort=False).review_body.apply(''.join).reset_index(name='all_review')

# # re combines and puts everything in lower case
# import re
# from tqdm import tqdm

# df_combined['all_review'] = df_combined['all_review'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

# def lower_case(input_str):
#     input_str = input_str.lower()
#     return input_str

# df_combined['all_review']= df_combined['all_review'].apply(lambda x: lower_case(x))

# df = df_combined

# df_sentences = df_combined.set_index("all_review")

# df_sentences = df_sentences["hotelName"].to_dict()
# df_sentences_list = list(df_sentences.keys())
# len(df_sentences_list)

# list(df_sentences.keys())[:5]
##
# df_sentences_list = [str(d) for d in tqdm(df_sentences_list)]


# # gives an encoding of the full concatenated list of corpus
# corpus = df_sentences_list
# corpus_embeddings = embedder.encode(corpus,show_progress_bar=True)

# paraphrases = util.paraphrase_mining(model, corpus)


# from sentence_transformers import SentenceTransformer, util
# import torch

# corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)


# with open("corpus.pkl" , "wb") as file1:
#   pkl.dump(corpus,file1)
# with open("corpus_embeddings.pkl" , "wb") as file2:
#   pkl.dump(corpus_embeddings,file2)
# with open("df.pkl" , "wb") as file3:
#   pkl.dump(df,file3)


# with open("corpus_embeddings.pkl" , "wb") as file_:
#   pkl.dump(corpus_embeddings,file_)
with open("df.pkl" , "rb") as file3:
    df = pkl.load(file3)
with open('corpus_embeddings.pkl', 'rb') as file2:
    corpus_embeddings = pkl.load(file2)
with open('corpus.pkl', 'rb') as file1:
    corpus = pkl.load(file1)
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)


# Query sentences:
userinput = st.text_input('What kind of hotel are you looking for?')
if not userinput:
    st.write("Nothing entered yet. Please enter what you're looking for.")
else:
    queries = [str(userinput)]
    query_embeddings = embedder.encode(queries,show_progress_bar=True)
##query testing


    def plot_cloud(wordcloud):
        plt.figure(figsize=(40, 30))
        # Display image
        plt.imshow(wordcloud) 
        # No axis details
        plt.axis("off");

    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(5, len(corpus))
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        st.write("\n\n======================\n\n")
        st.write("Query:", query)
        st.write("\nTop 5 most similar sentences in corpus:")

        for score, idx in zip(top_results[0], top_results[1]):
            st.write("(Score: {:.4f})".format(score))
            row_dict = df.loc[df['all_review']== corpus[idx]] 
            #writing in the column of row_dict that we want that excludes the index so only the hotelname remains. 
            st.write(row_dict['hotelName'].values[0] , "\n")
            #wordcloud = WordCloud(width= 3000, height = 1750, random_state=42, background_color='white', colormap='Pastel1', collocations=False, stopwords = STOPWORDS).generate(str(corpus[idx]))
            wordcloud = WordCloud(stopwords = stopwords).generate(corpus[idx])
            fig, ax = plt.subplots()
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot(fig)
            st.set_option('deprecation.showPyplotGlobalUse', False)
     
# import pickle as pkl
#upload a csv file, convereted that csv file after cleaning and converted to embedding
# if we dump the corpus into a pickle file, and then load the file and it will be saved in the system. 
# we havec saved the model, so we dont ahve to ever run the corpus 
# pkl.load()
# with open("/content/drive/MyDrive/BertSentenceSimilarity/Pickles/corpus_embeddings.pkl" , "wb") as file_:
#   pkl.dump(corpus_embeddings,file_)



