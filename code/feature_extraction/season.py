#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmcdonald
"""

import numpy as np
import datetime as dt
from time import mktime, strptime
from code.feature_extraction.feature_extractor import FeatureExtractor

class Season(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_charlength".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # 
    def _get_values(self, inputs):

        dates = []

        for date in inputs[0]:
            date = strptime(date, '%Y-%m-%d')
            date = dt.datetime.fromtimestamp(mktime(date))

            if date.month in [12, 1, 2]:
                dates.append(0)
            elif date.month in [3, 4, 5]:
                dates.append(2)
            elif date.month in [6, 7, 8]:
                dates.append(1)
            elif date.month in [9, 10, 11]:
                dates.append(3)

        result = np.asarray(dates)
        result = result.reshape(-1,1)
        return result