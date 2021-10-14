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
from code.preprocessing.stopwords import StopwordsRemover
from code.preprocessing.language_remover import LanguageRemover
from code.preprocessing.tokenizer import Tokenizer
from code.util import COLUMN_TWEET, SUFFIX_TOKENIZED, COLUMN_LANGUAGE

# setting up CLI
parser = argparse.ArgumentParser(description = "Various preprocessing steps")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output csv file")
parser.add_argument("-p", "--punctuation", action = "store_true", help = "remove punctuation")
parser.add_argument("-s", "--stopwords", action = "store_true", help = "remove stopwords")
parser.add_argument("-t", "--tokenize", action = "store_true", help = "tokenize given column into individual words")
#parser.add_argument("--tokenize_input", help = "input column to tokenize", default = 'output')
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
parser.add_argument("--language", help = "just use tweets with this language ", default = None)
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n",low_memory=False)

preprocess_col = 'preprocess_col'
# collect all preprocessors
preprocessors = []
if args.punctuation:
    preprocessors.append(PunctuationRemover("tweet", preprocess_col))
if args.stopwords:
    preprocessors.append(StopwordsRemover(preprocess_col, preprocess_col))
if args.tokenize:
    preprocessors.append(Tokenizer(preprocess_col, preprocess_col + SUFFIX_TOKENIZED))

# no need to detect languages, because it is already given
# if args.language is not None:
#   preprocessors.append(LanguageRemover())

if args.language is not None:
    # filter out one language
    before = len(df)
    df = df[df['language']==args.language]
    after = len(df)
    print("Filtered out: {0}".format(before-after))
    df.reset_index(drop=True, inplace=True)

# call all preprocessing steps
for preprocessor in preprocessors:
    df = preprocessor.fit_transform(df)

# drop useless line which makes problems with csv
del df['trans_dest\r']
# store the results
df.to_csv(args.output_file, index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")
#pdb.set_trace()
# create a pipeline if necessary and store it as pickle file
if args.export_file is not None:
    pipeline = make_pipeline(*preprocessors)
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)



