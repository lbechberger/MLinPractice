#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Given an input column, this class counts the length of the array per record
"""

import numpy as np
import ast
from src.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the character-based length as a feature
class CounterFE(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], f"{input_column}_count")


    # don't need to fit, so don't overwrite _set_variables()


    def _get_values(self, inputs):
        """
        Parses the string in every cell of the column/series as an array
        and counts the length in the cell of the output column
        """

        evaluated = inputs[0].apply(ast.literal_eval)
        result = np.array(evaluated.str.len())
        result = result.reshape(-1,1)
        return result
