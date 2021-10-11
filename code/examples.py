#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 09:20:09 2021

@author: ml
"""

###############################################################################
########################    DATA VISUALIZATION   ##############################
###############################################################################

# plotting with pandas
import csv
import pandas as pd

df = pd.read_csv("data/preprocessing/preprocessed.csv", quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

df["language"].value_counts().plot(kind = 'bar')
df["language"].value_counts().plot(kind = 'bar', logy = True)

df["date"] = df["date"].astype("datetime64")
df["label"].groupby(df["date"].dt.month).count().plot(kind = 'bar')


# plotting with matplotlib
import pickle
from matplotlib import pyplot as plt
import numpy as np

with open("data/feature_extraction/training.pickle", "rb") as f_in:
    data = pickle.load(f_in)

features = data["features"]
labels = data["labels"]

plt.hist(features)
plt.hist(features, range = [0,400])

pos = features[labels]
neg_index = np.array([not x for x in labels])
neg = features[neg_index]

bins = [0, 50, 100, 150, 200, 250, 300, 350, 400]

plt.hist(pos, bins = bins)
plt.hist(neg, bins = bins)

###############################################################################
########################    FEATURE EXTRACTION   ##############################
###############################################################################

# bigrams
import nltk
import string

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


###############################################################################
#####################    DIMENSIONALITY REDUCTION   ###########################
###############################################################################

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np

data_set = load_breast_cancer()
X = data_set.data
y = data_set.target
print("Data Set: ", X.shape, y.shape)
print("Combinatorics of binary feature values:", 2**30)


# PCA
print("\nPCA")
print('---')
pca = PCA(random_state = 42)
pca.fit(X)
print("explained variance (percentage): ", pca.explained_variance_ratio_)
print('most important component: ', pca.components_[0])
pca_transformed = pca.transform(X)
pca_transformed = pca_transformed[:,0:1]
print("after transformation: ", pca_transformed.shape, y.shape)
print("Compare: ", X[0], pca_transformed[0])


# wrapper
print("\nWrapper")
print("-------")

model = LogisticRegression(random_state = 42, max_iter = 10000)
rfe = RFE(model, n_features_to_select = 2)
rfe.fit(X,y)
print("Feature ranking according to RFE/LogReg:", rfe.ranking_)
index_of_first = np.where(rfe.ranking_ == 1)[0][0]
index_of_second = np.where(rfe.ranking_ == 2)[0][0]
print("Two most promising features: ", index_of_first, index_of_second)
wrapper_transformed = rfe.transform(X)
print("After transformation: ", wrapper_transformed.shape, y.shape)
print("compare: ", X[0], wrapper_transformed[0])


# Filter
print("\n Filter")
print("------")
skb = SelectKBest(score_func = mutual_info_classif, k = 3)
skb.fit(X,y)
print("Feature scores according to MI: ", skb.scores_)
filter_transformed  = skb.transform(X)
print("After transformation: ", filter_transformed.shape, y.shape)
print("Compare: ", X[0], filter_transformed[0])



# Embedded
print("\nEmbedded")
print("--------")
rf = RandomForestClassifier(n_estimators = 10, random_state=42)
rf.fit(X,y)
print("Feature imporance according to RF: ", rf.feature_importances_)
sfm = SelectFromModel(rf, threshold = 0.1, prefit = True)
embedded_transformed = sfm.transform(X)
print("After transformation: ", embedded_transformed.shape, y.shape)
print("Compare: ", X[0], embedded_transformed[0])

















