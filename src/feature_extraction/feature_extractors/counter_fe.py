#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Given an input column, this class counts the length of the array per record
"""

import numpy as np
import ast
from src.feature_extraction.feature_extractors.feature_extractor import FeatureExtractor

class CounterFE(FeatureExtractor):
    """
    Parses the string in every cell of the column/series as an list
    and counts the length in the cell of the output column
    """
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], f"{input_column}_count")


    # don't need to fit, so don't overwrite _set_variables()

    def _get_values(self, inputs):
        evaluated = inputs[0].apply(ast.literal_eval)
        result = np.array(evaluated.str.len())
        result = result.reshape(-1,1)
        return result
