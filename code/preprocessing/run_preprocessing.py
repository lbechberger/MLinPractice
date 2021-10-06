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
from code.preprocessing.tokenizer import Tokenizer
from code.preprocessing.lowercase import Lowercase
from code.util import SUFFIX_PUNCTUATION, SUFFIX_TOKENIZED, SUFFIX_LOWERCASED

# setting up CLI
parser = argparse.ArgumentParser(description = "Various preprocessing steps")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output csv file")
parser.add_argument("--pipeline", action='append', nargs='*', help="define a preprocessing pipeline e.g. --pipeline "
                                                                   "<column> preprocessor1 preprocessor 2 ... "
                                                                   "Available preprocessors: punctuation, "
                                                                   "tokenize")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

# collect all preprocessors
preprocessors = []
if args.pipeline:
    for pipeline in args.pipeline:
        current_column = ''
        for preprocessor in pipeline:
            if preprocessor == 'punctuation':
                preprocessors.append(PunctuationRemover(current_column, current_column+SUFFIX_PUNCTUATION))
                current_column = current_column+SUFFIX_PUNCTUATION
            elif preprocessor == 'tokenize':
                preprocessors.append(Tokenizer(current_column, current_column + SUFFIX_TOKENIZED))
                current_column = current_column + SUFFIX_TOKENIZED
            elif preprocessor == 'lowercase':
                preprocessors.append(Lowercase(current_column, current_column + SUFFIX_LOWERCASED))
                current_column = current_column + SUFFIX_LOWERCASED
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