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
COLUMN_LIKES = "likes_count"
COLUMN_RETWEETS = "retweets_count"

# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
COLUMN_PUNCTUATION = "tweet_no_punctuation"
COLUMN_TOKENIZED = "tweet_tokenized"
COLUMN_STOPWORDS = "tweet_no_stopwords"

<<<<<<< HEAD
SUFFIX_TOKENIZED = "_tokenized"
SUFFIX_STOPWORDS_REMOVED = "_stopwords_removed"

=======

DETECTION_MODEL = fasttext.load_model("code/models/lid.176.bin")

# to ignore a deprecation warning
warnings.simplefilter(action='ignore', category=FutureWarning)
fasttext.FastText.eprint = lambda x: None
>>>>>>> 2854caf (modified files and testing added)
