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
from code.feature_extraction.emoji_count import EmojiCount
from code.feature_extraction.hash_vector import HashVector
from code.feature_extraction.tfidf_vector import TfidfVector
from code.feature_extraction.feature_collector import FeatureCollector
from code.feature_extraction.hashtag_count import HashtagCount
from code.feature_extraction.photo_bool import PhotoBool
from code.feature_extraction.video_bool import VideoBool
from code.feature_extraction.word2vec import Word2Vec
from code.feature_extraction.time_feature import Hours
from code.util import (
    COLUMN_TWEET,
    COLUMN_LABEL,
    COLUMN_PREPROCESS,
    COLUMN_PHOTOS,
    COLUMN_REPLIES,
    COLUMN_VIDEO,
)


# setting up CLI
parser = argparse.ArgumentParser(description="Feature Extraction")
parser.add_argument("input_file", help="path to the input csv file")
parser.add_argument("output_file", help="path to the output pickle file")
parser.add_argument(
    "-e",
    "--export_file",
    help="create a pipeline and export to the given location",
    default=None,
)
parser.add_argument(
    "-i",
    "--import_file",
    help="import an existing pipeline from the given location",
    default=None,
)
parser.add_argument(
    "-c",
    "--char_length",
    action="store_true",
    help="compute the number of characters in the tweet",
)
parser.add_argument(
    "--hash_vec", action="store_true", help="compute the hash vector of the tweet"
)
parser.add_argument(
    "--tfidf_vec", action="store_true", help="compute the tf idf of the tweet"
)
parser.add_argument(
    "--photo_bool",
    action="store_true",
    help="tells whether the tweet contains photos or not",
)
parser.add_argument(
    "--video_bool",
    action="store_true",
    help="tells whether the tweet contains a video or not",
)
parser.add_argument(
    "--word2vec",
    action="store_true",
    help="compute the semantic distance of words to given keywords",
)
parser.add_argument(
    "--time", action="store_true", help="take into account what hour the tweet was sent"
)
parser.add_argument(
    "--emoji_count", action="store_true", help="count the emojis in a tweet"
)
parser.add_argument(
    "--hashtags", action="store_true", help="count hashtags of the tweet"
)

args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")

if args.import_file is not None:
    # simply import an exisiting FeatureCollector
    with open(args.import_file, "rb") as f_in:
        feature_collector = pickle.load(f_in)

else:  # need to create FeatureCollector manually

    # collect all feature extractors
    features = []
    if args.char_length:
        print("Add char_length feature")
        features.append(CharacterLength(COLUMN_PREPROCESS))
    if args.hash_vec:
        # hash of original tweet (without any changes)
        print("Add hash_vec feature")
        features.append(HashVector(COLUMN_TWEET))
    if args.hashtags:
        print("Add hashtags feature")
        # number of hashtags per tweet
        features.append(HashtagCount("hashtags"))
    if args.tfidf_vec:
        print("Add tfidf_vec feature")
        features.append(TfidfVector(COLUMN_PREPROCESS))
    if args.emoji_count:
        features.append(EmojiCount(COLUMN_TWEET))
    if args.photo_bool:
        print("Add photo_bool feature")
        # do photos exist or not
        features.append(PhotoBool(COLUMN_PHOTOS))
    if args.video_bool:
        print("Add video_bool feature")
        # does a video exist or not
        features.append(VideoBool(COLUMN_VIDEO))
    if args.word2vec:
        print("Add word2vec feature")
        features.append(Word2Vec("preprocess_col_tokenized"))
    if args.time:
        print("Add time feature")
        # when was the tweet published
        features.append(Hours("time"))

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

feature_names = feature_collector.get_feature_names(df)
# store the results
print("store the results")
results = {
    "features": feature_array,
    "labels": label_array,
    "feature_names": feature_names,
}

# replace 'NaN' in features
results["features"] = np.nan_to_num(results["features"])

with open(args.output_file, "wb") as f_out:
    pickle.dump(results, f_out)

# export the FeatureCollector as pickle file if desired by user
if args.export_file is not None:
    with open(args.export_file, "wb") as f_out:
        pickle.dump(feature_collector, f_out)

# use this if you want to inspect the produced output in a csv.
df_out = pd.DataFrame(feature_array, columns=feature_names)
df_out.to_csv("data/feature_extraction/features.csv")

# results.to_csv(args.output_file, index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")
