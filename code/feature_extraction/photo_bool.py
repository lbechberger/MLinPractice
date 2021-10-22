#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that tells whether photos are present or not.

Created on Wed Sep 29 12:29:25 2021

@author: shagemann
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the photo-bool as a feature
class PhotoBool(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_bool".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # 0 if no photos, return 1 else
    def _get_values(self, inputs):
        values = []
        for index, row in inputs[0].iteritems():
            if len(row) > 2:
                values.append(1)
            else:
                values.append(0)
        result = np.array(values)
        result = result.reshape(-1,1)
        return result
