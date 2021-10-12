#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import string
from code.preprocessing.preprocessor import Preprocessor
from langdetect import detect
from code.util import COLUMN_TWEET, COLUMN_LANGUAGE

class LanguageRemover(Preprocessor):
    
    # constructor
    def __init__(self, input_column = COLUMN_TWEET, output_column = COLUMN_LANGUAGE): #, language_to_keep = 'en'
        # input column "tweet", new output column
        super().__init__([input_column], output_column)
        #self.language_to_keep = language_to_keep
    
    # set internal variables based on input columns
    #def _set_variables(self, inputs):
        # store punctuation for later reference
        #self._punctuation = "[{}]".format(string.punctuation)
        #self.nlp = spacy.load('en')  # 1
        #self.nlp.add_pipe(LanguageDetector(), name='language_detector', last=True) #2

    
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        column = [detect(tweet) for tweet in inputs[0]]
        return column