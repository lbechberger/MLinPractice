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

class TfIdf(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_tfidf".format(input_column))
        self._vectorizer = TfidfVectorizer(input='content', max_features=200)

    # compute IDFs
    def _set_variables(self, inputs):
        return self._vectorizer.fit(inputs[0])

    # compute the tf-idf matrix
    def _get_values(self, inputs):
        print('TF-IDF vocabulary: {0}'.format(self._vectorizer.get_feature_names()))
        return self._vectorizer.transform(inputs[0]).toarray()