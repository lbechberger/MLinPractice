#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of preprocessing steps

Created on Tue Sep 28 16:43:18 2021

@author: lbechberger
"""

import argparse, csv, pickle
from re import L
import pandas as pd
from sklearn.pipeline import make_pipeline
from code.preprocessing.punctuation_remover import PunctuationRemover
from code.preprocessing.tokenizer import Tokenizer
from code.preprocessing.lemmatizer import Lemmatizer
from code.preprocessing.stopword_remover import Stopword_remover
from code.util import COLUMN_TWEET, SUFFIX_TOKENIZED, SUFFIX_LEMMATIZED, SUFFIX_REMOVED_STOPWORDS

# setting up CLI
parser = argparse.ArgumentParser(description = "Various preprocessing steps")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output csv file")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
parser.add_argument("--pipeline", action='append', nargs='*', help="define a preprocessing pipeline e.g. --pipeline "
                                                                   "<column> preprocessor1 preprocessor 2 ... "
                                                                   "Available preprocessors: punctuation, "
                                                                   "tokenize, lowercase, numbers, lemmatize, remove_stopwords")
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

# collect all preprocessors
preprocessors = []
if args.pipeline:
    for pipeline in args.pipeline:
        current_column = ''
        for preprocessor in pipeline:
            if preprocessor == 'tokenize':
                preprocessors.append(Tokenizer(current_column, current_column + SUFFIX_TOKENIZED))
                current_column = current_column + SUFFIX_TOKENIZED
            elif preprocessor == 'lemmatize':
                preprocessors.append(Lemmatizer(current_column, current_column+SUFFIX_LEMMATIZED))
                current_column = current_column + SUFFIX_LEMMATIZED
            elif preprocessor == 'remove_stopwords':
                preprocessors.append(Stopword_remover(current_column, current_column+SUFFIX_REMOVED_STOPWORDS))
                current_column = current_column + SUFFIX_REMOVED_STOPWORDS
            else:
                # first argument in pipeline is column
                current_column = preprocessor

# call all preprocessing steps
for preprocessor in preprocessors:
    df = preprocessor.fit_transform(df)

# store the results
df.to_csv(args.output_file, index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")

# create a pipeline if necessary and store it as pickle file
if args.export_file is not None:
    pipeline = make_pipeline(*preprocessors)
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)