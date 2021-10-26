#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Console-based application for tweet classification.

Created on Wed Sep 29 14:49:25 2021

@author: lbechberger, mkalcher, magmueller, shagemann
"""

import argparse, pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import make_pipeline
from code.util import COLUMN_TWEET, COLUMN_HASHTAGS, COLUMN_PHOTOS, COLUMN_VIDEO, COLUMN_TIME

# setting up CLI
# the user can choose the features he wants to extract in the file 'feature_extraction.sh'
parser = argparse.ArgumentParser(description = "Application")
parser.add_argument("preprocessing_file", help = "path to the pickle file containing the preprocessing")
parser.add_argument("feature_file", help = "path to the pickle file containing the feature extraction")
parser.add_argument("classifier_file", help = "path to the pickle file containing the classifier")
args = parser.parse_args()

# load all the pipeline steps
# dimensionality_reduction is NOT used here
with open(args.preprocessing_file, 'rb') as f_in:
    preprocessing = pickle.load(f_in)
with open(args.feature_file, 'rb') as f_in:
    feature_extraction = pickle.load(f_in)
with open(args.classifier_file, 'rb') as f_in:
    classifier = pickle.load(f_in)

# chain them together into a single pipeline
pipeline = make_pipeline(preprocessing, feature_extraction, classifier)

# headline output
print("Welcome to ViralTweeter v0.1!")
print("-----------------------------")
print("")

while True:
    # ask user for input
    tweet = input("Please type in your tweet (type 'quit' to quit the program): ")

    # terminate if necessary
    if tweet == "quit":
        print("Okay, goodbye!")
        break

    # ask if the tweet contains videos
    check_bool = True
    while(check_bool == True):
        input_video = input("How many videos does your tweet contain? (type in 0, 1, 2, ...): ")
        if not input_video.isnumeric():
            print("Your input must be an integer!")
        else:
            check_bool = False

    # check how many photos the tweet contains
    input_photos = []
    for word in tweet.split():
        if "https://pbs.twimg.com/media" in word and (".png" in word or ".jpg" in word):
            input_photos.append(word)

    # get current time
    now = datetime.now()

    # if not terminated: create pandas DataFrame and put it through the pipeline
    # the feature columns that are not generated in feature extraction need to be put in manually
    # the video column needs to be manually put in by the user
    # the current time is being generated automatically
    df = pd.DataFrame()
    df[COLUMN_TWEET] = [tweet]
    df[COLUMN_HASHTAGS] = [tweet.split()]
    df[COLUMN_PHOTOS] = [input_photos]
    df[COLUMN_VIDEO] = [input_video]
    df[COLUMN_TIME] = [now.strftime("%H:%M:%S")]
    prediction = pipeline.predict(df)
    try:
        confidence = pipeline.predict_proba(df)
        print("Prediction: {0}, Confidence: {1}".format(prediction, confidence))
    except:
        print("Prediction:", prediction)
        if prediction.flat[0] == False:
            print("Your tweet will most likely not go viral.")
        elif prediction.flat[0] == True:
            print("Your tweet will most likely go viral.")

    print("")
