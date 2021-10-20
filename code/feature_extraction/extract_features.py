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
from code.feature_extraction.hash_vector import HashVector
from code.feature_extraction.tfidf_vector import TfidfVector
from code.feature_extraction.feature_collector import FeatureCollector
from code.feature_extraction.photo_bool import PhotoBool
from code.feature_extraction.video_bool import VideoBool
from code.feature_extraction.replies_count import RepliesCount
from code.feature_extraction.word2vec import Word2Vec
from code.feature_extraction.time_feature import Hours
from code.util import COLUMN_TWEET, COLUMN_LABEL, COLUMN_PREPROCESS, COLUMN_PHOTOS, COLUMN_REPLIES, COLUMN_VIDEO


# setting up CLI
parser = argparse.ArgumentParser(description = "Feature Extraction")
parser.add_argument("input_file", help="path to the input csv file")
parser.add_argument("output_file", help="path to the output pickle file")
parser.add_argument("-e", "--export_file", 
		    help="create a pipeline and export to the given location", default=None)
parser.add_argument("-i", "--import_file", 
		    help="import an existing pipeline from the given location", default=None)
parser.add_argument("-c", "--char_length", action="store_true", 
		    help="compute the number of characters in the tweet")
parser.add_argument("--hash_vec", action="store_true", 
		    help="compute the hash vector of the tweet")
parser.add_argument("--tfidf_vec", action="store_true", 
		    help="compute the tf idf of the tweet")
parser.add_argument("--photo_bool", action="store_true", 
		    help="tells whether the tweet contains photos or not")
parser.add_argument("--video_bool", action="store_true", 
		    help="tells whether the tweet contains a video or not")
parser.add_argument("--replies_count", action="store_true", 
		    help="compute the amount of replies of the tweet")
parser.add_argument("--word2vec", action="store_true", 
		    help="compute the semantic distance of words to given keywords")
parser.add_argument("--time", action="store_true", 
		    help="take into account what hour the tweet was sent")
parser.add_argument("--tfidf_vec", action="store_true", 
		    help="take into account what hour the tweet was sent")

args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting=csv.QUOTE_NONNUMERIC,
                 lineterminator="\n")
df = df[0:1000]

df = df[0:1000]

if args.import_file is not None:
    # simply import an exisiting FeatureCollector
    with open(args.import_file, "rb") as f_in:
        feature_collector = pickle.load(f_in)

else:    # need to create FeatureCollector manually

    # collect all feature extractors
    features = []
    if args.char_length:

        features.append(CharacterLength(COLUMN_PREPROCESS))
    if args.hash_vec:

        features.append(HashVector(COLUMN_PREPROCESS))
    if args.tfidf_vec:

        features.append(TfidfVector(COLUMN_PREPROCESS))
    if args.photo_bool:
        # do photos exist or not
        features.append(PhotoBool(COLUMN_PHOTOS))
    if args.video_bool:
        # does a video exist or not
        features.append(VideoBool(COLUMN_VIDEO))
    if args.replies_count:
        # how many replies does the tweet have
        features.append(RepliesCount(COLUMN_REPLIES))
    if args.word2vec:
        features.append(Word2Vec('preprocess_col_tokenized'))
    if args.time:
        # how many replies does the tweet have
        features.append(Hours('time'))


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
