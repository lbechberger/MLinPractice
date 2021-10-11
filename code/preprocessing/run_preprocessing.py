#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of preprocessing steps

Created on Tue Sep 28 16:43:18 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
import re
from sklearn.pipeline import make_pipeline
from code.preprocessing.punctuation_remover import PunctuationRemover
from code.preprocessing.tokenizer import Tokenizer
from code.preprocessing.lowercase import Lowercase
from code.preprocessing.standardize import Standardizer
from code.preprocessing.expand import Expander
from code.preprocessing.prune_languages import LanguagePruner
from code.preprocessing.regex_replacer import RegexReplacer
from code.preprocessing.lemmatizer import Lemmatizer
from code.preprocessing.stopword_remover import Stopword_remover
from code.util import SUFFIX_PUNCTUATION, SUFFIX_STANDARDIZED, SUFFIX_TOKENIZED, SUFFIX_LOWERCASED, SUFFIX_URLS_REMOVED, SUFFIX_NUMBERS_REPLACED, TOKEN_NUMBER, SUFFIX_CONTRACTIONS, SUFFIX_LEMMATIZED, SUFFIX_REMOVED_STOPWORDS


# setting up CLI
parser = argparse.ArgumentParser(description = "Various preprocessing steps")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output csv file")
parser.add_argument("-l", "--prune_lang", action="store_true")
parser.add_argument("--pipeline", action='append', nargs='*', help="define a preprocessing pipeline e.g. --pipeline "
                                                                   "<column> preprocessor1 preprocessor 2 ... "
                                                                   "IMPORTANT: remove_urls has to run before punctuation"
                                                                   "Available preprocessors: remove_urls, "
                                                                   "lowercase, expand, tokenize, punctuation, "
                                                                   "numbers, standardize, lemmatize, remove_stopwords")

parser.add_argument("--fast", action = "store_true", help = "only run preprocessing on a subset of the data set")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

# Comment in for testing
if args.fast:
    df = df.drop(labels = range(1000, df.shape[0]), axis = 0)

# Removes rows in a language other than the one specified to keep
if args.prune_lang:

    language_pruner = LanguagePruner(df)
    language_pruner.get_language_counts()
    df = language_pruner.drop_rows_by_language(language = "en")

# collect all preprocessors
preprocessors = []
if args.pipeline:
    for pipeline in args.pipeline:
        current_column = ''
        for preprocessor in pipeline:
            if preprocessor == 'remove_urls':
                preprocessors.append(RegexReplacer(current_column, current_column + SUFFIX_URLS_REMOVED, r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', ""))
                current_column = current_column + SUFFIX_URLS_REMOVED
            elif preprocessor == 'punctuation':
                preprocessors.append(PunctuationRemover(current_column, current_column + SUFFIX_PUNCTUATION))
                current_column = current_column + SUFFIX_PUNCTUATION
            elif preprocessor == 'lowercase':
                preprocessors.append(Lowercase(current_column, current_column + SUFFIX_LOWERCASED))
                current_column = current_column + SUFFIX_LOWERCASED
            elif preprocessor == 'expand':
                preprocessors.append(Expander(current_column, current_column + SUFFIX_CONTRACTIONS,))
                current_column = current_column + SUFFIX_CONTRACTIONS
            elif preprocessor == 'tokenize':
                preprocessors.append(Tokenizer(current_column, current_column + SUFFIX_TOKENIZED))
                current_column = current_column + SUFFIX_TOKENIZED
            elif preprocessor == 'numbers':
                preprocessors.append(RegexReplacer(current_column, current_column + SUFFIX_NUMBERS_REPLACED, r'(?<=\W)\d+(?=\W)|^\d+(?=\W)|(?<=\W)\d+$', TOKEN_NUMBER))
                current_column = current_column + SUFFIX_NUMBERS_REPLACED
            elif preprocessor == 'standardize':
                preprocessors.append(Standardizer(current_column, current_column + SUFFIX_STANDARDIZED))
                current_column = current_column + SUFFIX_STANDARDIZED
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