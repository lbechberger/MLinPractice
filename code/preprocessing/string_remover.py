#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that removes punctuation from the original tweet text.
Created on Wed Sep 29 09:45:56 2021
@author: lbechberger
"""
import string
from code.preprocessing.preprocessor import Preprocessor
from code.util import COLUMN_TWEET, COLUMN_PUNCTUATION
from nltk.corpus import stopwords
import pandas as pd

STOPWORDS = set(stopwords.words('english'))

# removes punctuation from the original tweet
# inspired by https://stackoverflow.com/a/45600350
class StringRemover(Preprocessor):
    
    # constructor
    def __init__(self, inputcol, outputcol):
        # input column "tweet", new output column
        super().__init__([inputcol], outputcol)
    
    # set internal variables based on input columns
    #def _set_variables(self, inputs):
        # store punctuation for later reference
        # self._punctuation = "[{}]".format(string.punctuation)
        
    
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        column = inputs[0].str
        
        # replace stopwords with empty string
        # column = [' '.join([word for word in tweet if word.lower() not in STOPWORDS]) for tweet in column.split()]
        
        # replace links with empty string
        # column = [' '.join([word for word in tweet if word.startswith('https') is False]) for tweet in column.split()]
        
        # replace emojis with empty string
        column = [' '.join([word for word in tweet if str(word.encode('unicode-escape').decode('ASCII')).__contains__('\\') is False]) for tweet in column.split()] 
        
        column = pd.Series(column)

        return column