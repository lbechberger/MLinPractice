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
COLUMN_URLS = "urls"
COLUMN_HASHTAGS = "hashtags"
COLUMN_RETWEET_BOOL = "retweet"
COLUMN_VIDEO = "video"
COLUMN_REPLY_TO = "reply_to"
COLUMN_MENTIONS = "mentions"
COLUMN_PHOTOS = "photos"
COLUMN_DATE = "date"
COLUMN_TIME = "time"
COLUMN_PREPROCESSED_TWEET = "tweet_urls_removed_lowercased_expanded_no_punctuation_standardized_tokenized_numbers_replaced_lemmatized_removed_stopwords"
COLUMN_UNTOKENIZED_TWEET = "tweet_urls_removed_lowercased_expanded_no_punctuation_standardized"

# column names of novel columns for preprocessing
COLUMN_LABEL = "label"

SUFFIX_PUNCTUATION = "_no_punctuation"
SUFFIX_TOKENIZED = "_tokenized"
SUFFIX_LOWERCASED = "_lowercased"
SUFFIX_URLS_REMOVED = "_urls_removed"
SUFFIX_NUMBERS_REPLACED = "_numbers_replaced"
SUFFIX_STANDARDIZED = "_standardized"
SUFFIX_CONTRACTIONS = "_expanded"
SUFFIX_REMOVED_STOPWORDS = "_removed_stopwords"
SUFFIX_TOKENIZED = "_tokenized"
SUFFIX_LEMMATIZED = "_lemmatized"

#Special tokens
TOKEN_NUMBER = "__NUMBER__"
