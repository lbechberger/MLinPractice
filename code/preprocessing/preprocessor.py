#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Superclass for all preprocessors.

Created on Tue Sep 28 17:06:35 2021

@author: lbechberger
"""

from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessor(BaseEstimator,TransformerMixin):
    """Inheritance from BaseEstimator and TransformerMixin from the sklearn library."""    
    
    
    def __init__(self, input_columns, output_column):
        """
        Constructor
        calls super Constructor from BaseEstimator and TransformerMixin (initializes them)
        sets the respective _output_column and _input_columns
        """
        super(BaseEstimator, self).__init__()
        super(TransformerMixin, self).__init__()
        self._output_column = output_column
        self._input_columns = input_columns
    

    def _set_variables(self, inputs):
        """
        Set the internal variables based on input columns.
        Needs to be implemented by subclass       
        """
        pass
    

    def fit(self, df):
        """Fit function: takes pandas DataFrame to set any internal variables"""
        inputs = []
        
        # collect all input columns from the dataframe
        for input_col in self._input_columns:
            inputs.append(df[input_col])
        
        # call _set_variables 
        self._set_variables(inputs)
        
        return self
    

    def _get_values(self, inputs):
        """
        Get preprocessed column based on the inputs from the DataFrame 
        and internal variables.
        Needs to be implemented by subclass.
        """
        pass
        

    def transform(self, df):
        """
        Transform function: transforms pandas DataFrame 
        based on any internal variables.
        """
        inputs = []
        # collect all input columns from the dataframe
        for input_col in self._input_columns:
            inputs.append(df[input_col])
        
        # add to copy of DataFrame
        df_copy = df.copy()
        df_copy[self._output_column] = self._get_values(inputs)  
        
        return df_copy