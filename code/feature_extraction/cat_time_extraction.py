#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature that gives categorical value to datetime of post

@author: lmcdonald
"""

import numpy as np
import datetime as dt
from time import mktime, strptime
from sklearn.preprocessing import OneHotEncoder
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting a time feature
# options: weekday, month, season or time of day of the post
class CatTimeExtractor(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column, feature):
        super().__init__([input_column], "{1}".format(input_column, feature))
        # what feature is to be computed
        self._feature = feature
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the respective time feature
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
                    # posted in winter
                    result.append(0)
                elif datetime.month in [3, 4, 5]:
                    # posted in spring
                    result.append(2)
                elif datetime.month in [6, 7, 8]:
                    # posted in summer
                    result.append(1)
                elif datetime.month in [9, 10, 11]:
                    # posted in fall
                    result.append(3)

            elif self._feature == "daytime":

                datetime = strptime(datetime, '%H:%M:%S')
                datetime = dt.datetime.fromtimestamp(mktime(datetime))

                if datetime.hour in range(0,6):
                    # posted at night
                    result.append(0)
                elif datetime.hour in range(6,12):
                    # posted in morning
                    result.append(1)
                elif datetime.hour in range(12,18):
                    # posted in afternoon
                    result.append(2)
                elif datetime.hour in range(18,24):
                    # posted in evening
                    result.append(3)

        onehot_encoder = OneHotEncoder(sparse=False)
        result = np.asarray(result)
        result2 = result
        result = result.reshape(len(result), 1)
        result = onehot_encoder.fit_transform(result)
        
        return result
