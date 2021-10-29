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
from code.util import COLUMN_TWEET, COLUMN_TOKENIZED, COLUMN_PUNCTUATION, COLUMN_STOPWORDS, COLUMN_LANGUAGE, COLUMN_LEMMATIZED

# setting up CLI
<<<<<<< HEAD
parser = argparse.ArgumentParser(description="Various preprocessing steps")
parser.add_argument("input_file", help="path to the input csv file")
parser.add_argument("output_file", help="path to the output csv file")
parser.add_argument("-p", "--punctuation",
                    action="store_true", help="remove punctuation")
parser.add_argument("-t", "--tokenize", action="store_true",
                    help="tokenize given column into individual words")
parser.add_argument("--tokenize_input",
                    help="input column used for tokenization", default=COLUMN_TWEET)
parser.add_argument("-e", "--export_file",
                    help="create a pipeline and export to the given location", default=None)
=======
parser = argparse.ArgumentParser(description = "Various preprocessing steps")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output csv file")
parser.add_argument("-t", "--tokenize", action = "store_true", help = "tokenize given column into individual words")
parser.add_argument("--tokenize_input", help = "input column to tokenize", default = COLUMN_TWEET)
parser.add_argument("-p", "--punctuation_removing", action = "store_true", help = "remove punctuation")
parser.add_argument("--punctuation_removing_input", help = "input column to stopword_remover", default = COLUMN_TOKENIZED)
parser.add_argument("-sw", "--stopwords_removing", action = "store_true", help = "remove stopwords from the given column")
parser.add_argument("--stopwords_removing_input", help = "input column to stopword_remover", default = [COLUMN_PUNCTUATION, COLUMN_LANGUAGE])
parser.add_argument("-l", "--lemmatize", action = "store_true", help = "extract the lemmas from the given column")
parser.add_argument("--lemmatize_input", help = "input column to lemmatizer", default = COLUMN_STOPWORDS)
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
>>>>>>> 3021bf2 (create a stopwords remover)
args = parser.parse_args()

# load data
<<<<<<< HEAD
df = pd.read_csv(args.input_file, quoting=csv.QUOTE_NONNUMERIC,
                 lineterminator="\n")
=======
df = pd.read_csv(args.input_file, low_memory=False, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")
>>>>>>> 2854caf (modified files and testing added)

# collect all preprocessors
preprocessors = []
if args.tokenize:
<<<<<<< HEAD
<<<<<<< HEAD
    preprocessors.append(Tokenizer(args.tokenize_input,
                         args.tokenize_input + SUFFIX_TOKENIZED))
=======
    preprocessors.append(Tokenizer(args.tokenize_input, args.tokenize_input + SUFFIX_TOKENIZED))
if args.stopwords_remover:
    preprocessors.append(StopwordsRemover(args.tokenize_input, args.tokenize_input + SUFFIX_STOPWORDS_REMOVED))
>>>>>>> 3021bf2 (create a stopwords remover)
=======
    preprocessors.append(Tokenizer(args.tokenize_input, COLUMN_TOKENIZED))
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
if args.punctuation_remover:
    preprocessors.append(PunctuationRemover(args.punctuation_remover_input, COLUMN_PUNCTUATION))
if args.stopwords_remover:
    preprocessors.append(StopwordsRemover(args.stopwords_remover_input, COLUMN_STOPWORDS))
>>>>>>> 2854caf (modified files and testing added)
=======
=======
>>>>>>> 7105dbc (resolve the conflict)
=======
>>>>>>> 828d2d3 (fix merge issues)
if args.punctuation_removing:
    preprocessors.append(PunctuationRemover(args.punctuation_removing_input, COLUMN_PUNCTUATION))
if args.stopwords_removing:
    preprocessors.append(StopwordsRemover(args.stopwords_removing_input, COLUMN_STOPWORDS))
if args.lemmatize:
    preprocessors.append(Lemmatizer(args.lemmatize_input, COLUMN_LEMMATIZED))
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 0898e45 (add the lemmatizer and its test)
=======
=======
if args.punctuation_remover:
    preprocessors.append(PunctuationRemover(args.punctuation_remover_input, COLUMN_PUNCTUATION))
if args.stopwords_remover:
    preprocessors.append(StopwordsRemover(args.stopwords_remover_input, COLUMN_STOPWORDS))
>>>>>>> 49c39fa (resolve the conflict)
>>>>>>> 7105dbc (resolve the conflict)
=======
>>>>>>> 828d2d3 (fix merge issues)

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
