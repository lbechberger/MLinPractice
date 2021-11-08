#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that represents the attached media file as integer

Created on Fri Oct 29 12:33:04 2021

@author: ldankert
"""

import numpy as np
import pandas as pd
from src.util import COLUMN_PHOTOS, COLUMN_VIDEO
from src.feature_extraction.feature_extractor import FeatureExtractor


# class for extracting the character-based length as a feature
class MediaType(FeatureExtractor):

    # constructor
    def __init__(self, input_columns):
        super().__init__(input_columns, "media_type")

    # returns 3 columns, one for each media type (Photo, Video, None)
    def _get_values(self, inputs):
        result = []
        for photo, video in zip(inputs[0],inputs[1]):
            if photo != "[]":
                result.append([1,0,0])
            elif video == 1:
                result.append([0,1,0])
            else:
                result.append([0,0,1])
        return np.array(result)

