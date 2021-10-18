#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that tells how many replies a tweet has.

Created on Wed Sep 29 12:29:25 2021

@author: shagemann
"""

import numpy as np
from codes.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the photo-bool as a feature
class RepliesCount(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_replies_count".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # use the replies count column as a feature
    def _get_values(self, inputs):
        values = []
        for index, row in inputs[0].iteritems():
            values.append(int(row))
        result = np.array(values)
        result = result.reshape(-1,1)
        return result
