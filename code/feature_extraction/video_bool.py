#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that tells whether a video is in the tweet or not.

Created on Wed Sep 29 12:29:25 2021

@author: shagemann
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the photo-bool as a feature
class VideoBool(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_bool".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # use 'video' column as a feature
    # 0 if no video, return 1 else
    def _get_values(self, inputs):
        values = []
        for index, row in inputs[0].iteritems():
                values.append(int(row))
        result = np.array(values)
        result = result.reshape(-1,1)
        return result
