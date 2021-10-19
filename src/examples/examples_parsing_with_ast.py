#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: parsing a double quoted list of strings
"""

"""
string wrapping around a list of strings parsed to a list of strings
"""

import csv
import ast
import pandas as pd

df = pd.read_csv("data/preprocessing/preprocessed.csv", quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
tokenized_string = df["tweet_tokenized"][0]
tokenized_list = ast.literal_eval(tokenized_string)