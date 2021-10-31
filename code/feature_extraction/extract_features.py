#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of feature extractors.

Created on Wed Sep 29 11:00:24 2021

@author: lbechberger
"""

import argparse
import csv
import pickle
import pandas as pd
import numpy as np
from code.feature_extraction.character_length import CharacterLength
from code.feature_extraction.bigrams import BigramFeature
from code.feature_extraction.images_counter import ImagesCounter
from code.feature_extraction.videos_counter import VideosCounter
from code.feature_extraction.popular_hashtags import PopularHashtags
from code.feature_extraction.feature_collector import FeatureCollector
from code.feature_extraction.sentiment_analysis import SentimentAnalysis
from code.feature_extraction.mentions_counter import MentionsCounter
from code.feature_extraction.retweets_counter import RetweetsCounter
from code.feature_extraction.likes_counter import LikesCounter
from code.feature_extraction.replies_counter import RepliesCounter
from code.feature_extraction.similar_tweets import SimilarTweets
from code.util import COLUMN_REPLIES, COLUMN_HASHTAGS, COLUMN_MENTIONS, COLUMN_TWEET, COLUMN_LABEL, COLUMN_STEMMED, COLUMN_IMAGES, COLUMN_VIDEOS, COLUMN_RETWEET, COLUMN_LIKES


# setting up CLI
parser = argparse.ArgumentParser(description="Feature Extraction")
parser.add_argument("input_file", help="path to the input csv file")
parser.add_argument("output_file", help="path to the output pickle file")
parser.add_argument("-e", "--export_file",
                    help="create a pipeline and export to the given location", default=None)
parser.add_argument("-i", "--import_file",
                    help="import an existing pipeline from the given location", default=None)
parser.add_argument("-b", "--bigram", action="store_true",
                    help="get the bigrams")
parser.add_argument("-st", "--similar_tweets", action="store_true",
                    help="check the similarity between the tweets")
parser.add_argument("-c", "--char_length", action="store_true",
                    help="compute the number of characters in the tweet")
parser.add_argument("-p", "--photos_shared", action="store_true",
                    help="compute the number of photos attached to the tweet")
parser.add_argument("-v", "--videos_shared", action="store_true",
                    help="compute the number of videos attached to the tweet")
parser.add_argument("-ph", "--popular_hashtags", action="store_true",
                    help="compute the number of hashtags attached to the tweet")
parser.add_argument("-sa", "--sentiment_analysis", action="store_true",
                    help="check the sentiment analyses")
parser.add_argument("-m", "--mentions_counter", action="store_true",
                    help="count the mentions")
parser.add_argument("-r", "--retweet_num", action="store_true",
                    help="count the retweets")
parser.add_argument("-l", "--likes_num", action="store_true",
                    help="count the likes")
parser.add_argument("-re", "--replies_num", action="store_true",
                    help="count the replies number")

args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting=csv.QUOTE_NONNUMERIC,
                 lineterminator="\n")

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
    if args.bigram:
        features.append(BigramFeature(COLUMN_STEMMED))
    if args.photos_shared:
        features.append(ImagesCounter(COLUMN_IMAGES))
    if args.videos_shared:
        features.append(VideosCounter(COLUMN_VIDEOS))
    if args.popular_hashtags:
        features.append(PopularHashtags(COLUMN_HASHTAGS))
    if args.sentiment_analysis:
        features.append(SentimentAnalysis(COLUMN_STEMMED))
    if args.mentions_counter:
        features.append(MentionsCounter(COLUMN_MENTIONS))
    if args.retweet_num:
        features.append(RetweetsCounter(COLUMN_RETWEET))
    if args.likes_num:
        features.append(LikesCounter(COLUMN_LIKES))
    if args.replies_num:
        features.append(RepliesCounter(COLUMN_REPLIES))
    if args.similar_tweets:
        features.append(SimilarTweets(COLUMN_STEMMED))

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
