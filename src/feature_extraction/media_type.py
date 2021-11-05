#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that represents the attached media file as integer

Created on Fri Oct 29 12:33:04 2021

@author: ldankert
"""

import numpy as np
from src.feature_extraction.feature_extractor import FeatureExtractor


# class for extracting the character-based length as a feature
class MediaType(FeatureExtractor):

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "media_type")


    # returns 3 columns, one for each media type
    def _get_values(self, inputs):
        result = []
        for media in np.array(inputs[0]):
            if media == "Photo":
                media_number = [1,0,0]
            elif media == "Video":
                media_number = [0,1,0]
            else:
                media_number = [0,0,1]
            result.append(media_number)
        result = np.array(result)
        return result
