#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility file for collecting frequently used constants and helper functions.

Created on Wed Sep 29 10:50:36 2021

@author: lbechberger
"""

# column names for the original data frame
COLUMN_TWEET = "tweet"
COLUMN_LIKES = "likes_count"
COLUMN_RETWEETS = "retweets_count"
COLUMN_PHOTOS = "photos"
COLUMN_VIDEO = "video"
COLUMN_REPLIES = "replies_count"

# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
COLUMN_PUNCTUATION = "tweet_no_punctuation"
COLUMN_LANGUAGE = "language"
COLUMN_PREPROCESS = 'preprocess_col'
SUFFIX_TOKENIZED = "_tokenized"


# split dataset (all in one)
TEST_SIZE = 0.015
# number of features for hash vector
HASH_VECTOR_N_FEATURES = 2**20
HASH_VECTOR_N_FEATURES_TITLE = 2**3
NGRAM_RANGE = (1, 2)

# classifier 
KNN_K = 5
MAX_ITER_LOGISTIC = 50000
MAX_ITER_LINEAR_SVC = 5000
MAX_ITER_SGD = 10000
ALPHA_SGD = 1e-6
