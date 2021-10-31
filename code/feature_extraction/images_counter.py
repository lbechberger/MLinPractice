#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of images in the given column.

Created on Wed Sep 29 12:29:25 2021

@author: lbechberger
"""
import ast
import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor


class ImagesCounter(FeatureExtractor):

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_shared".format(input_column))

    def _get_values(self, inputs):
        result = np.array(inputs[0].apply(lambda x: len(ast.literal_eval(x))))
        result = result.reshape(-1, 1)
        return result
