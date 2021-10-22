#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of feature extractors.

Created on Wed Sep 29 11:00:24 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
import numpy as np
from code.feature_extraction.character_length import CharacterLength
from code.feature_extraction.feature_collector import FeatureCollector
from code.feature_extraction.month import MonthExtractor
from code.feature_extraction.sentiment import SentimentAnalyzer
from code.feature_extraction.photos import Photos
from code.feature_extraction.mention import Mentions
from code.feature_extraction.retweets import RetweetExtractor
from code.feature_extraction.url import URL
from code.feature_extraction.replies import RepliesExtractor
from code.feature_extraction.hashtags import Hashtags
from code.util import COLUMN_TWEET, COLUMN_LABEL, COLUMN_MONTH, COLUMN_PHOTOS, COLUMN_MENTIONS, COLUMN_URL, COLUMN_RETWEETS, COLUMN_REPLIES, COLUMN_HASHTAG


# setting up CLI
parser = argparse.ArgumentParser(description = "Feature Extraction")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output pickle file")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import an existing pipeline from the given location", default = None)

# <--- Features --->
parser.add_argument("-c", "--char_length", action = "store_true", help = "compute the number of characters in the tweet")
parser.add_argument("-m", "--month", action = "store_true", help = "extract month in which tweet was published")
parser.add_argument("-s", "--sentiment", action = "store_true", help = "extracts compound sentiment from original tweet")
parser.add_argument("-p", "--photos", action = "store_true", help = "extracts binary for whether tweet has photo(s) attached")
parser.add_argument("-@", "--mention", action = "store_true", help = "extracts binary for whether someone has been mentioned by the tweet author")
parser.add_argument("-u", "--url", action = "store_true", help = "extracts binary for whether a url is attached to tweet")
parser.add_argument("-r", "--retweet", action = "store_true", help = "extracts number of retweets")
parser.add_argument("-k", "--replies", action = "store_true", help = "extracts number of replies")
parser.add_argument("-o", "--hashtag", action = "store_true", help = "extracts binary for whether a hashtag is attached to tweet")

args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

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
        
    if args.month:
        # month in which tweet was published
        features.append(MonthExtractor(COLUMN_MONTH))
        
    if args.sentiment:
        # compound sentiment of tweet
        features.append(SentimentAnalyzer(COLUMN_TWEET))
        
    if args.photos:
        # photos attached to original tweet
        features.append(Photos(COLUMN_PHOTOS))
        
    if args.mention:
        # mentions contained in tweet
        features.append(Mentions(COLUMN_MENTIONS))
    
    if args.url:
        # url attached to tweet
       features.append(URL(COLUMN_URL))
       
    if args.retweet:
        # number of retweets
        features.append(RetweetExtractor(COLUMN_RETWEETS))
    
    if args.replies:
        # number of replies
        features.append(RepliesExtractor(COLUMN_REPLIES))
        
    if args.hashtag:
        # number of replies
        features.append(Hashtags(COLUMN_HASHTAG))
        
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