#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of characters in the given column.

Created on Wed Sep 29 12:29:25 2021

@author: lbechberger
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor
from sklearn.feature_extraction.text import TfidfVectorizer


# class for extracting the character-based length as a feature


class TfidfVector(FeatureExtractor):

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column],
                         "{0}_tfidfvector".format(input_column))

    # don't need to fit, so don't overwrite _set_variables()

    # compute the word length based on the inputs
    def _get_values(self, inputs):
        # inputs is list of text documents
        # create the transform

        vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 2))
        # encode document
        vector = vectorizer.fit_transform(inputs[0])
        
        return vector.toarray()
