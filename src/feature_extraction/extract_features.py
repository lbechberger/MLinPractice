#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of feature extractors.
"""

import argparse, csv, pickle
import pandas as pd
import numpy as np
from src.feature_extraction import character_length
from src.feature_extraction.character_length import CharacterLengthFE
from src.feature_extraction.counter_fe import CounterFE
from src.feature_extraction.feature_collector import FeatureCollector
from src.feature_extraction.sentiment_fe import SentimentFE
from src.util import COLUMN_MENTIONS, COLUMN_PHOTOS, COLUMN_TWEET
from src.util import COLUMN_LABEL, COLUMN_HASHTAGS , COLUMN_URLS
from src.util import COLUMN_CASHTAGS, COLUMN_REPLY_TO, COLUMN_TWEET_TOKENIZED


def main():
    # setting up CLI
    parser = argparse.ArgumentParser(description = "Feature Extraction")
    parser.add_argument("input_file", help = "path to the input csv file")
    parser.add_argument("output_file", help = "path to the output pickle file")
    parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
    parser.add_argument("-i", "--import_file", help = "import an existing pipeline from the given location", default = None)    
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

    is_feature_collector_provided = args.import_file is not None
    
    if is_feature_collector_provided:
        feature_collector = get_feature_collector_from_file(args.import_file)
    else:
        feature_collector = create_and_fit_feature_collector(df)

    # apply the given FeatureCollector on the current data set
    # maps the pandas DataFrame to an numpy array
    feature_array = feature_collector.transform(df)

    # get label array
    label_array = np.array(df[COLUMN_LABEL])
    label_array = label_array.reshape(-1, 1)

    # print("features\n ---\n", feature_array)

    # store the results
    results = {"features": feature_array, "labels": label_array, 
            "feature_names": feature_collector.get_feature_names()}
    with open(args.output_file, 'wb') as f_out:
        pickle.dump(results, f_out)

    # export the FeatureCollector as pickle file if desired by user
    if args.export_file is not None:
        with open(args.export_file, 'wb') as f_out:
            pickle.dump(feature_collector, f_out)

def get_feature_collector_from_file(filepath):
    with open(filepath, "rb") as f_in:
        feature_collector = pickle.load(f_in)
    return feature_collector
    
def create_and_fit_feature_collector(df: pd.DataFrame):

    featureExtractors = instantiate_feature_extractors()
    feature_collector = FeatureCollector(featureExtractors)
    feature_collector.fit(df) # assumed to be training data

    return feature_collector

def instantiate_feature_extractors():
    featureExtractors = []

    featureExtractors.append(CharacterLengthFE(COLUMN_TWEET))
    featureExtractors.append(SentimentFE(COLUMN_TWEET))

    count_columns = [
        COLUMN_MENTIONS,
        COLUMN_PHOTOS,
        COLUMN_HASHTAGS,
        COLUMN_URLS,
        COLUMN_CASHTAGS,
        COLUMN_REPLY_TO,
        COLUMN_TWEET_TOKENIZED
    ]
    for count_columns in count_columns:
        featureExtractors.append(CounterFE(count_columns))

    return  featureExtractors

if __name__ == "__main__":
    main()