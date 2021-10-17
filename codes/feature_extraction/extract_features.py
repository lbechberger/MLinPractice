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
from codes.feature_extraction.character_length import CharacterLength
from codes.feature_extraction.hash_vector import HashVector
from codes.feature_extraction.feature_collector import FeatureCollector
from codes.feature_extraction.photo_bool import PhotoBool
from codes.feature_extraction.replies_count import RepliesCount
from codes.util import COLUMN_TWEET, COLUMN_LABEL, COLUMN_PREPROCESS, COLUMN_PHOTOS, COLUMN_REPLIES


# setting up CLI
parser = argparse.ArgumentParser(description = "Feature Extraction")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output pickle file")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import an existing pipeline from the given location", default = None)
parser.add_argument("-c", "--char_length", action = "store_true", help = "compute the number of characters in the tweet")
parser.add_argument("--hash_vec", action = "store_true", help = "compute the hash vector of the tweet")
parser.add_argument("--photo_bool", action= "store_true", help= "tells whether the tweet contains photos or not")
parser.add_argument("--replies_count", action="store_true", help="compute the amount of replies of the tweet")
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
    if args.hash_vec:
        # hash of original tweet (without any changes)
        features.append(HashVector(COLUMN_TWEET))
    if args.photo_bool:
        # do photos exist or not
        features.append(PhotoBool(COLUMN_PHOTOS))
    if args.replies_count:
        # how many replies does the tweet have
        features.append(RepliesCount(COLUMN_REPLIES))

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
