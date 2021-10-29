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
<<<<<<< HEAD
<<<<<<< HEAD
COLUMN_STOPWORDS = "tweet_no_stopwords"
<<<<<<< HEAD
<<<<<<< HEAD

<<<<<<< HEAD
SUFFIX_TOKENIZED = "_tokenized"
SUFFIX_STOPWORDS_REMOVED = "_stopwords_removed"

=======
=======
=======
=======
>>>>>>> 828d2d3 (fix merge issues)
COLUMN_STOPWORDS = "tweet_no_stopwords"

>>>>>>> 7105dbc (resolve the conflict)

# to ignore a deprecation warning
warnings.simplefilter(action='ignore', category=FutureWarning)
fasttext.FastText.eprint = lambda x: None
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 2854caf (modified files and testing added)
=======
>>>>>>> fe92cf0 (fixed error in the stopword remover)
=======
COLUMN_STOPWORDS = "tweet_no_stopwords"
>>>>>>> 0898e45 (add the lemmatizer and its test)
=======
>>>>>>> 49c39fa (resolve the conflict)
>>>>>>> 7105dbc (resolve the conflict)
=======
>>>>>>> 828d2d3 (fix merge issues)
