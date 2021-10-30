#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that removes punctuation from the original tweet text.

Created on Wed Sep 29 09:45:56 2021

@author: lbechberger
modified by dhesenkamp
"""

import string
from code.preprocessing.preprocessor import Preprocessor


class PunctuationRemover(Preprocessor):
    """
    Class to remove punctuation marks from given input
    inspired by https://stackoverflow.com/a/45600350
    """
    
    
    def __init__(self, input_column, output_column):
        """Constuctor, calls super Constructor"""
        super().__init__([input_column], output_column)
    
    
    def _set_variables(self, inputs):
        """
        Stores punctuation for later reference
        """
        self._punctuation = "[{}]".format(string.punctuation)
    
    
    def _get_values(self, inputs):
        """
        Replaces a punctuation mark with an empty string
        """
        column = inputs[0].str.replace(self._punctuation, "")
        
        return column