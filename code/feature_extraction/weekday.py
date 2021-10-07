#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of characters in the given column.

@author: lmcdonald
"""

import numpy as np
import datetime as dt
from time import mktime, strptime
from code.feature_extraction.feature_extractor import FeatureExtractor

class Weekday(FeatureExtractor):
    
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
            dates.append(date.weekday())
            print(date.weekday())

        result = np.asarray(dates)

        print(result)

        result = np.array(inputs[0].str.len())
        result = result.reshape(-1,1)
        return result
