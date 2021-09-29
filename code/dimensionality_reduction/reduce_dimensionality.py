#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply a dimensionality reduction technique.

Created on Wed Sep 29 13:33:37 2021

@author: lbechberger
"""

import argparse, pickle
import numpy as np


# setting up CLI
parser = argparse.ArgumentParser(description = "Dimensionality reduction")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("output_file", help = "path to the output pickle file")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import an existing pipeline from the given location", default = None)
args = parser.parse_args()

with open(args.input_file, 'rb') as f_in:
    input_data = pickle.load(f_in)

features = input_data["features"]