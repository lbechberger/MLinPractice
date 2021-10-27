#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 22:05:32 2021

@author: maximilian
"""

import numpy as np
import pandas as pd
from code.feature_extraction.feature_extractor import FeatureExtractor


class Hours(FeatureExtractor):
    """Create a feature for the hour of creation."""

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_hours".format(input_column))

    def _get_values(self, inputs):
        hours = pd.to_datetime(inputs[0], format="%H:%M:%S").dt.hour
        result = np.array(hours)
        result = result.reshape(-1, 1)
        return result
