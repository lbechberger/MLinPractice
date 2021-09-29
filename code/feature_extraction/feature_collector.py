#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collects the feature values from many different feature extractors.

Created on Wed Sep 29 12:36:01 2021

@author: lbechberger
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# extend FeatureExtractor for the sake of simplicity
class FeatureCollector(FeatureExtractor):
    
    # constructor
    def __init__(self, features):
        
        # store features
        self._features = features
        
        # collect input columns
        input_columns = []
        for feature in self._features:
            input_columns += feature.get_input_columns()
        
        # remove duplicate columns
        input_colums = list(set(input_columns))
        
        # call constructor of super class
        super().__init__(input_columns, "FeatureCollector")

    
    # overwrite fit: instead of calling _set_variables(), we forward the call to the features
    def fit(self, df):
        
        for feature in self._features:
            feature.fit(df)

    # overwrite transform: instead of calling _get_values(), we forward the call to the features
    def transform(self, df):
        
        all_feature_values = []
        
        for feature in self._features:
            all_feature_values.append(feature.transform(df))
        
        result = np.concatenate(all_feature_values, axis = 1)
        return result

    def get_feature_names(self):
        feature_names = []
        for feature in self._features:
            feature_names.append(feature.get_feature_name())
        return feature_names