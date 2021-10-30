#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of characters in the given column.

Created on Wed Sep 29 12:29:25 2021

@author: lbechberger
"""
import ast
import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the character-based length as a feature


class Images(FeatureExtractor):

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_attached".format(input_column))

    # don't need to fit, so don't overwrite _set_variables()

    # compute the word length based on the inputs
    def _get_values(self, inputs):
        result = np.array(inputs[0].apply(lambda x: len(ast.literal_eval(x))))
        result = result.reshape(-1, 1)
        return result
