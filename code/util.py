#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility file for collecting frequently used constants and helper functions.
Created on Wed Sep 29 10:50:36 2021
@author: lbechberger
"""

# column names for the original data frame
COLUMN_TWEET = "tweet"
COLUMN_LANGUAGE = "language"
COLUMN_LIKES = "likes_count"
COLUMN_RETWEETS = "retweets_count"
COLUMN_LEMMATIZED = "tweet_lemmatized"
COLUMN_STEMMED = "tweet_stemmed"

# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
COLUMN_PUNCTUATION = "tweet_no_punctuation"
COLUMN_TOKENIZED = "tweet_tokenized"
COLUMN_STOPWORDS = "tweet_no_stopwords"

# column names of novel columns for feature extracting
COLUMN_UNIQUE_BIGRAMS = "tweet_unique_bigrams"
COLUMN_IMAGES = "photos"
COLUMN_VIDEOS = "video"
COLUMN_HASHTAGS = "hashtags"
COLUMN_MENTIONS = "mentions"
COLUMN_RETWEET = "retweets_count"
COLUMN_LIKES = "likes_count"
COLUMN_REPLIES = "replies_count"
