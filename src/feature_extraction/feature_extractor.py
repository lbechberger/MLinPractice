#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class for all of our feature extractors.

Created on Wed Sep 29 12:22:13 2021

@author: lbechberger
"""

from sklearn.base import BaseEstimator, TransformerMixin

# base class for all feature extractors
#   inherits from BaseEstimator (as pretty much everything in sklearn)
#       and TransformerMixin (allowing for fit, transform, and fit_transform methods)
class FeatureExtractor(BaseEstimator,TransformerMixin):
    
    # constructor
    def __init__(self, input_columns, feature_name):
        super(BaseEstimator, self).__init__()
        super(TransformerMixin, self).__init__()
        self._input_columns = input_columns
        self._feature_name = feature_name
        
    # access to feature name
    def get_feature_name(self):
        return self._feature_name
    
    # access to input colums
    def get_input_columns(self):
        return self._input_columns

    
    # set internal variables based on input columns
    # to be implemented by subclass!
    def _set_variables(self, inputs):
        pass
    
    # fit function: takes pandas DataFrame to set any internal variables
    def fit(self, df):
        
        inputs = []
        # collect all input columns from df
        for input_col in self._input_columns:
            inputs.append(df[input_col])
        
        # call _set_variables (to be implemented by subclass)
        self._set_variables(inputs)
        
        return self
    
         
    # get feature values based on input column and internal variables
    # should return a numpy array
    # to be implemented by subclass!
    def _get_values(self, inputs):
        pass
        
    # transform function: transforms pandas DataFrame to numpy array of feature values
    def transform(self, df):

        inputs = []
        # collect all input columns from df
        for input_col in self._input_columns:
            inputs.append(df[input_col])
            
        return self._get_values(inputs)