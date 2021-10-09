#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmcdonald
"""

import numpy as np
import ast
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the character-based length as a feature
class BooleanCounter(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column, count_type = "boolean"):
        super().__init__([input_column], "{0}_{1}".format(input_column, count_type))
        
        # specifies if count is absolute or boolean (> 1 or 0)
        self._count_type = count_type
        self._input_column = input_column
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the word length based on the inputs
    def _get_values(self, inputs):

        result = []

        for row in inputs[0]:
            if self._count_type == "boolean":
                result.append(int(bool(row)))
            elif self._count_type == "count":
                result.append(len(ast.literal_eval(row)))
 
        result = np.asarray(result)

        result = result.reshape(-1,1)
        return result