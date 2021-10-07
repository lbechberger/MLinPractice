#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmcdonald
"""

import numpy as np
import datetime as dt
from time import mktime, strptime
from code.feature_extraction.feature_extractor import FeatureExtractor

class Daytime(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_charlength".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # 
    def _get_values(self, inputs):

        times = []

        for time in inputs[0]:
            time = strptime(time, '%H:%M:%S')
            time = dt.datetime.fromtimestamp(mktime(time))

            if time.hour in range(0,6):
                times.append(0)
            elif time.hour in range(6,12):
                times.append(1)
            elif time.hour in range(12,18):
                times.append(2)
            elif time.hour in range(18,24):
                times.append(3)

        result = np.asarray(times)
        result = result.reshape(-1,1)
        return result