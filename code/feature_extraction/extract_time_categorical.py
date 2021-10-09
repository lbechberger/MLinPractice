#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmcdonald
"""

import numpy as np
import datetime as dt
from time import mktime, strptime
from code.feature_extraction.feature_extractor import FeatureExtractor

class CatTimeExtractor(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column, feature):
        super().__init__([input_column], "{1}".format(input_column, feature))
        self._feature = feature
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # 
    def _get_values(self, inputs):

        result = []

        for datetime in inputs[0]:

            if self._feature == "month":

                datetime = strptime(datetime, '%Y-%m-%d')
                datetime = dt.datetime.fromtimestamp(mktime(datetime))
                result.append(datetime.month)

            elif self._feature == "weekday":

                datetime = strptime(datetime, '%Y-%m-%d')
                datetime = dt.datetime.fromtimestamp(mktime(datetime))
                result.append(datetime.weekday())

            elif self._feature == "season":

                datetime = strptime(datetime, '%Y-%m-%d')
                datetime = dt.datetime.fromtimestamp(mktime(datetime))

                if datetime.month in [12, 1, 2]:
                    result.append(0)
                elif datetime.month in [3, 4, 5]:
                    result.append(2)
                elif datetime.month in [6, 7, 8]:
                    result.append(1)
                elif datetime.month in [9, 10, 11]:
                    result.append(3)

            elif self._feature == "daytime":

                datetime = strptime(datetime, '%H:%M:%S')
                datetime = dt.datetime.fromtimestamp(mktime(datetime))

                if datetime.hour in range(0,6):
                    result.append(0)
                elif datetime.hour in range(6,12):
                    result.append(1)
                elif datetime.hour in range(12,18):
                    result.append(2)
                elif datetime.hour in range(18,24):
                    result.append(3)

        result = np.asarray(result)

        print ("example result: ", self._feature, result)
        result = result.reshape(-1,1)
        return result
