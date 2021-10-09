#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Named-entity-recognition (NER): Feature that recognizes named entities in the tweet.
Created on Sat Oct 9 23:59:37 2021
@author: louiskhub
"""

import numpy as np
import nltk
from nltk import pos_tag
from code.feature_extraction.feature_extractor import FeatureExtractor


class NER(FeatureExtractor):
    """class for recognition of named entities in the tweet"""

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_sentiment".format(input_column))

    # don't need to fit, so don't overwrite _set_variables()

    # compute the sentiment based on the inputs
    def _get_values(self, inputs):
        

        return 