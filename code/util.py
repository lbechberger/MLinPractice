#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility file for collecting frequently used constants and helper functions.

Created on Wed Sep 29 10:50:36 2021

@author: lbechberger
"""
import fasttext, warnings

# column names for the original data frame
COLUMN_TWEET = "tweet"
COLUMN_LANGUAGE = "language"
COLUMN_LIKES = "likes_count"
COLUMN_RETWEETS = "retweets_count"
COLUMN_LEMMATIZED = "lemmatized"
COLUMN_STEMMED = "stemmed"

# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
COLUMN_PUNCTUATION = "tweet_no_punctuation"
COLUMN_TOKENIZED = "tweet_tokenized"
COLUMN_STOPWORDS = "tweet_no_stopwords"


# to ignore a deprecation warning
warnings.simplefilter(action='ignore', category=FutureWarning)
fasttext.FastText.eprint = lambda x: None
