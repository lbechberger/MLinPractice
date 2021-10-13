#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Console-based application for tweet classification.

Created on Wed Sep 29 14:49:25 2021

@author: lbechberger
"""

import argparse, pickle
import pandas as pd
from sklearn.pipeline import make_pipeline
from code.util import COLUMN_TWEET

# setting up CLI
parser = argparse.ArgumentParser(description = "Application")
parser.add_argument("preprocessing_file", help = "path to the pickle file containing the preprocessing")
parser.add_argument("feature_file", help = "path to the pickle file containing the feature extraction")
parser.add_argument("dim_red_file", help = "path to the pickle file containing the dimensionality reduction")
parser.add_argument("classifier_file", help = "path to the pickle file containing the classifier")
args = parser.parse_args()

# load all the pipeline steps
with open(args.preprocessing_file, 'rb') as f_in:
    preprocessing = pickle.load(f_in)
with open(args.feature_file, 'rb') as f_in:
    feature_extraction = pickle.load(f_in)
with open(args.dim_red_file, 'rb') as f_in:
    dimensionality_reduction = pickle.load(f_in)
with open(args.classifier_file, 'rb') as f_in:
    classifier = pickle.load(f_in)["classifier"]

# chain them together into a single pipeline
pipeline = make_pipeline(preprocessing, feature_extraction, dimensionality_reduction, classifier)

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
    
    # if not terminated: create pandas DataFrame and put it through the pipeline
    df = pd.DataFrame()
    df[COLUMN_TWEET] = [tweet]
    
    prediction = pipeline.predict(df)
    confidence = pipeline.predict_proba(df)
    
    print("Prediction: {0}, Confidence: {1}".format(prediction, confidence))
    print("")
    
