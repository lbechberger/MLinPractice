#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of feature extractors.

Created on Wed Sep 29 11:00:24 2021

@author: lbechberger
"""

import argparse, csv, pickle
from typing import Counter
import pandas as pd
import numpy as np
from code.feature_extraction.character_length import CharacterLength
from code.feature_extraction.count_boolean import BooleanCounter
from code.feature_extraction.tf_idf import TfIdf
from code.feature_extraction.threads import Threads
from code.feature_extraction.feature_collector import FeatureCollector
from code.feature_extraction.sentiment import Sentiment
from code.util import COLUMN_HASHTAGS, COLUMN_MENTIONS, COLUMN_PHOTOS, COLUMN_REPLY_TO, COLUMN_RETWEET_BOOL, COLUMN_TWEET, COLUMN_LABEL, COLUMN_URLS, COLUMN_VIDEO


# setting up CLI
parser = argparse.ArgumentParser(description = "Feature Extraction")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output pickle file")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import an existing pipeline from the given location", default = None)
parser.add_argument("--char_length", action = "store_true", help = "compute the length  in the tweet")
parser.add_argument("--hashtag_count", action = "store_true", help = "compute the number of hashtags in the tweet")
parser.add_argument("--mentions_count", action = "store_true", help = "compute the number of mentions in the tweet")
parser.add_argument("--reply_to_count", action = "store_true", help = "compute the number of accounts replied to in the tweet")
parser.add_argument("-p", "--photos_count", action = "store_true", help = "compute the number of photos in the tweet")
parser.add_argument("-u", "--url_count", action = "store_true", help = "compute the number of URLs used in the tweet")
parser.add_argument("--item_count", action = "store_true", help = "compute the absolute count of items, else compute boolean if items > 0")
parser.add_argument("-v", "--video_binary", action = "store_true", help = "compute the binary of if the tweet is a video")
parser.add_argument("-r", "--retweet_binary", action = "store_true", help = "compute the binary of if the tweet is a retweet")
parser.add_argument("-t", "--tfidf", action = "store_true", help = "compute word-wise tf-idf")
parser.add_argument("-s", "--sentiment", action = "store_true", help = "compute the tweet sentiment")
parser.add_argument("--threads", action = "store_true", help = "match tweets that are part of a thread")

args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

if args.item_count:
    count_type = "count"
else:
    count_type = "boolean"

if args.import_file is not None:
    # simply import an exisiting FeatureCollector
    with open(args.import_file, "rb") as f_in:
        feature_collector = pickle.load(f_in)

else:    # need to create FeatureCollector manually

    # collect all feature extractors
    features = []
    if args.char_length:
        # character length of original tweet (without any changes)
        features.append(CharacterLength(COLUMN_TWEET))
    if args.hashtag_count:
        # number (or if) hashtags used
        features.append(BooleanCounter(COLUMN_HASHTAGS, count_type))
    if args.mentions_count:
        # number (or if) mentions used
        features.append(BooleanCounter(COLUMN_MENTIONS, count_type))
    if args.reply_to_count:
        # number (or if) reply_to used
        features.append(BooleanCounter(COLUMN_REPLY_TO, count_type))
    if args.photos_count:
        # number (or if) photos used
        features.append(BooleanCounter(COLUMN_PHOTOS, count_type))
    if args.url_count:
        # number (or if) URLs used
        features.append(BooleanCounter(COLUMN_URLS, count_type))
    if args.video_binary:
        # convert if tweet contains video to boolean
        features.append(BooleanCounter(COLUMN_VIDEO, "boolean"))
    if args.retweet_binary:
        # convert if tweet is retweet to boolean
        features.append(BooleanCounter(COLUMN_RETWEET_BOOL, "boolean"))
    if args.tfidf:
        features.append(TfIdf('tweet_urls_removed_no_punctuation_lowercased_expanded_tokenized_numbers_replaced_standardized_lemmatized_removed_stopwords'))
    if args.sentiment:
        # sentiment of original tweet (without any changes)
        features.append(Sentiment(COLUMN_TWEET))
    if args.threads:
        # character length of original tweet (without any changes)
        features.append(Threads(COLUMN_TWEET))
    
    # create overall FeatureCollector
    feature_collector = FeatureCollector(features)
    
    # fit it on the given data set (assumed to be training data)
    feature_collector.fit(df)


# apply the given FeatureCollector on the current data set
# maps the pandas DataFrame to an numpy array
feature_array = feature_collector.transform(df)

# get label array
label_array = np.array(df[COLUMN_LABEL])
label_array = label_array.reshape(-1, 1)

# store the results
results = {"features": feature_array, "labels": label_array, 
           "feature_names": feature_collector.get_feature_names()}
with open(args.output_file, 'wb') as f_out:
    pickle.dump(results, f_out)

# export the FeatureCollector as pickle file if desired by user
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(feature_collector, f_out)