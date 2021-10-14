#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of preprocessing steps

Created on Tue Sep 28 16:43:18 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
from sklearn.pipeline import make_pipeline
from code.preprocessing.punctuation_remover import PunctuationRemover
from code.preprocessing.stopwords_remover import StopwordsRemover
from code.preprocessing.tokenizer import Tokenizer
from code.util import COLUMN_TWEET, COLUMN_TOKENIZED, COLUMN_PUNCTUATION, COLUMN_STOPWORDS

# setting up CLI
parser = argparse.ArgumentParser(description = "Various preprocessing steps")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output csv file")
parser.add_argument("-t", "--tokenize", action = "store_true", help = "tokenize given column into individual words")
parser.add_argument("--tokenize_input", help = "input column to tokenize", default = COLUMN_TWEET)
parser.add_argument("-p", "--punctuation_remover", action = "store_true", help = "remove punctuation")
parser.add_argument("--punctuation_remover_input", help = "input column to stopword_remover", default = COLUMN_TOKENIZED)
parser.add_argument("-sw", "--stopwords_remover", action = "store_true", help = "remove stopwords from the given column")
parser.add_argument("--stopwords_remover_input", help = "input column to stopword_remover", default = COLUMN_PUNCTUATION)
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, low_memory=False, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

# collect all preprocessors
preprocessors = []
if args.tokenize:
    preprocessors.append(Tokenizer(args.tokenize_input, COLUMN_TOKENIZED))
if args.punctuation_remover:
    preprocessors.append(PunctuationRemover(args.punctuation_remover_input, COLUMN_PUNCTUATION))
if args.stopwords_remover:
    preprocessors.append(StopwordsRemover(args.stopwords_remover_input, COLUMN_STOPWORDS))

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