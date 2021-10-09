#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmcdonald
"""

import numpy as np
import pandas as pd
import ast
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the character-based length as a feature
class ItemCounter(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column, count_type = "boolean"):
        super().__init__([input_column], "{0}_{1}".format(input_column, count_type))
        self._count_type = count_type
        self._input_column = input_column
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the word length based on the inputs
    def _get_values(self, inputs):

        result = []

        for ls in inputs[0]:
            if self._count_type == "boolean":
                result.append(int(bool(ast.literal_eval(ls))))
            elif self._count_type == "count":
                result.append(len(ast.literal_eval(ls)))
 
        result = np.asarray(result)
        print("example result of", self._input_column, result)
        result = result.reshape(-1,1)
        return result