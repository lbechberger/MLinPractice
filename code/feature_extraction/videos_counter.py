#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of videos in the given column.

Created on Wed Sep 29 12:29:25 2021

@author: rfarah
"""
import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor


class VideosCounter(FeatureExtractor):

    def __init__(self, input_column):
        super().__init__([input_column], "{0}_shared".format(input_column))


    # compute the video list length based on the inputs
    def _get_values(self, inputs):
        result = np.array(inputs[0])
        result = result.reshape(-1, 1)
        return result
