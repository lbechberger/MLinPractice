#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Splits the preprocessed data into training, validation, and test set.

Created on Tue Sep 28 16:45:51 2021

@author: lbechberger
"""

import os, argparse, csv
import pandas as pd
from sklearn.model_selection import train_test_split
from code.util import COLUMN_LABEL

# setting up CLI
parser = argparse.ArgumentParser(description = "Splitting the data set")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_folder", help = "path to the output folder")
parser.add_argument("-s", '--seed', type = int, help = "seed for the random number generator", default = None)
parser.add_argument("-t", '--test_size', type = float, help = "relative size of the test set", default = 0.2)
parser.add_argument("-v", '--validation_size', type = float, help = "relative size of the validation set", default = 0.2)
args = parser.parse_args()

# load the data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

# split into (training & validation) and test set
X, X_test = train_test_split(df, test_size = args.test_size, random_state = args.seed, shuffle = True, stratify = df[COLUMN_LABEL])

# split remainder into training and validation
relative_validation_size = args.validation_size / (1 - args.test_size)
X_train, X_val = train_test_split(X, test_size = relative_validation_size, random_state = args.seed, shuffle = True, stratify = X[COLUMN_LABEL])

# store the three data sets separately
X_train.to_csv(os.path.join(args.output_folder, "training.csv"), index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")
X_val.to_csv(os.path.join(args.output_folder, "validation.csv"), index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")
X_test.to_csv(os.path.join(args.output_folder, "test.csv"), index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")

print("Training: {0} examples, Validation: {1} examples, Test: {2} examples".format(len(X_train), len(X_val), len(X_test)))