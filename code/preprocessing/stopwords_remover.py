#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that removes stopwords from the original tweet text depending on the language.

Created on Wed Sep 29 09:45:56 2021

@author: rfarah
"""
import spacy
from langdetect import detect
from code.preprocessing.preprocessor import Preprocessor
from code.util import COLUMN_TWEET, COLUMN_STOPWORDS


class StopwordsRemover(Preprocessor):
    # constructor
    def __init__(self):
        # input column "tweet", new output column
        super().__init__([COLUMN_TWEET], COLUMN_STOPWORDS)

    # set internal variables based on input columns
    def _set_variables(self, inputs):
        # store punctuation for later reference
        if detect(inputs) == "fr":
            self._stopwords = spacy.lang.fr.stop_words.STOP_WORDS
        elif detect(inputs) == "de":
            self._stopwords = spacy.lang.de.stop_words.STOP_WORDS
        elif detect(inputs) == "es":
            self._stopwords = spacy.lang.es.stop_words.STOP_WORDS
        elif detect(inputs) == "ar":
            self._stopwords = spacy.lang.ar.stop_words.STOP_WORDS
        else:
            self._stopwords = spacy.lang.en.stop_words.STOP_WORDS
    
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        # replace punctuation with empty string
        # warnings.simplefilter(action='ignore', category=FutureWarning)
        column = inputs[0].str.replace(self._stopwords, "")
        return column