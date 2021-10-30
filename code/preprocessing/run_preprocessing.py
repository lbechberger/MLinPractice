#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of preprocessing steps
Created on Tue Sep 28 16:43:18 2021
@author: lbechberger
"""

import argparse
import csv
import pickle
import pandas as pd
from sklearn.pipeline import make_pipeline
from code.preprocessing.lemmatizer import Lemmatizer
from code.preprocessing.punctuation_remover import PunctuationRemover
from code.preprocessing.stopwords_remover import StopwordsRemover
from code.preprocessing.tokenizer import Tokenizer
from code.preprocessing.stemmer import Stemmer
from code.util import COLUMN_TWEET, COLUMN_TOKENIZED, COLUMN_PUNCTUATION, COLUMN_STOPWORDS, COLUMN_LANGUAGE, COLUMN_LEMMATIZED, COLUMN_STEMMED

# setting up CLI
parser = argparse.ArgumentParser(description="Various preprocessing steps")
parser.add_argument("input_file", help="path to the input csv file")
parser.add_argument("output_file", help="path to the output csv file")
parser.add_argument("-t", "--tokenize", action="store_true",
                    help="tokenize given column into individual words")
parser.add_argument("--tokenize_input",
                    help="input column to tokenize", default=COLUMN_TWEET)
parser.add_argument("-p", "--punctuation_removing",
                    action="store_true", help="remove punctuation")
parser.add_argument("--punctuation_removing_input",
                    help="input column to stopword_remover", default=COLUMN_TOKENIZED)
parser.add_argument("-sw", "--stopwords_removing", action="store_true",
                    help="remove stopwords from the given column")
parser.add_argument("--stopwords_removing_input", help="input column to stopword_remover",
                    default=[COLUMN_PUNCTUATION, COLUMN_LANGUAGE])
parser.add_argument("-s", "--stemming", action="store_true",
                    help="stemm the given column")
parser.add_argument("--stemming_input", help="input column to stemmer",
                    default=[COLUMN_STOPWORDS, COLUMN_LANGUAGE])
parser.add_argument("-l", "--lemmatize", action="store_true",
                    help="extract the lemmas from the given column")
parser.add_argument("--lemmatize_input",
                    help="input column to lemmatizer", default=COLUMN_STOPWORDS)
parser.add_argument("-e", "--export_file",
                    help="create a pipeline and export to the given location", default=None)
parser.add_argument("-d", "--debug_limit",
                    help="limit data points to be preprocessed to given number", default=None)
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, low_memory=False,
                 quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")

# collect all preprocessors
preprocessors = []
if args.tokenize:
    preprocessors.append(Tokenizer(args.tokenize_input, COLUMN_TOKENIZED))
if args.punctuation_removing:
    preprocessors.append(PunctuationRemover(
        args.punctuation_removing_input, COLUMN_PUNCTUATION))
if args.stopwords_removing:
    preprocessors.append(StopwordsRemover(
        args.stopwords_removing_input, COLUMN_STOPWORDS))
if args.lemmatize:
    preprocessors.append(Lemmatizer(args.lemmatize_input, COLUMN_LEMMATIZED))
if args.stemming:
    preprocessors.append(Stemmer(args.stemming_input, COLUMN_STEMMED))

# check if data point number should be limited for debug purposes
if (args.debug_limit is not None) and (int(args.debug_limit) > 0):
    print("running with limited amount of datapoints: " + str(args.debug_limit))
    df = df.iloc[:int(args.debug_limit)]

# call all preprocessing steps
for preprocessor in preprocessors:
    df = preprocessor.fit_transform(df)

# store the results
df.to_csv(args.output_file, index=False,
          quoting=csv.QUOTE_NONNUMERIC, line_terminator="\n")

# create a pipeline if necessary and store it as pickle file
if args.export_file is not None:
    pipeline = make_pipeline(*preprocessors)
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)
