#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Some examples for feature extraction:
- bigrams
- TF-IDF
- WordNet synonyms
- word embeddings from gensim word2vec-google-news-300 model
- one hot encoding of categorical data
  - day and night are similar in nature, but on the opposite ends of the array)
"""

# bigrams
import nltk
import string
import csv
import pandas as pd

df = pd.read_csv("data/preprocessing/preprocessed.csv", quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

text = "John Wilkes Booth shot Abraham Lincoln. Abraham Lincoln was not shot inside the White House."
tokens = nltk.word_tokenize(text)
tokens = [token for token in tokens if token not in string.punctuation]

bigrams = nltk.bigrams(tokens)
freq_dist = nltk.FreqDist(bigrams)
freq_list = []
for bigram, freq in freq_dist.items():
    freq_list.append([bigram, freq])
freq_list.sort(key = lambda x: x[1], reverse = True)
for i in range(len(freq_list)):
    print(freq_list[i])


# tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tweets = df["tweet"][:100]
vectorizer = TfidfVectorizer()
tf_idf_vectors = vectorizer.fit_transform(tweets).todense()

print(tf_idf_vectors.shape)
print(vectorizer.get_feature_names()[142:145])
print(tf_idf_vectors[66:71, 142:145])

tf_idf_similarities = cosine_similarity(tf_idf_vectors)
print(tf_idf_similarities[:5,:5])


# NER
text = "John Wilkes Booth shot Abraham Lincoln. Abraham Lincoln was not shot inside the White House."
sentences = nltk.sent_tokenize(text)
for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    pos_tagged = nltk.pos_tag(words)
    ne_chunked = nltk.ne_chunk(pos_tagged)
    print(ne_chunked)


# WordNet
dog_synsets = nltk.corpus.wordnet.synsets('dog')
for syn in dog_synsets:
    words = [str(lemma.name()) for lemma in syn.lemmas()]
    print(syn, words, syn.definition(), syn.hypernyms())
    print("")


# word2vec
import gensim.downloader as api

embeddings = api.load('word2vec-google-news-300')
pairs = [('car', 'minivan'), ('car', 'airplane'), ('car', 'cereal')]

for w1, w2 in pairs:
    print("{0} - {1}: {2}".format(w1, w2, embeddings.similarity(w1, w2)))

dog_vector = embeddings['dog']


# one hot encoding
from sklearn.preprocessing import OneHotEncoder
import numpy as np

features = np.array([["morning"], ["afternoon"], ["evening"], ["night"], ["afternoon"]])
encoder = OneHotEncoder(sparse = False)
encoder.fit(features)
encoder.transform(features)
