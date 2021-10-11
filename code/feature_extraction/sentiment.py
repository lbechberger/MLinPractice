#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of characters in the given column.

Created on Wed Sep 29 12:29:25 2021

@author: lbechberger
"""

import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from code.feature_extraction.feature_extractor import FeatureExtractor


class Sentiment(FeatureExtractor):
    """class for extracting the tweet sentiment as a feature"""

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_sentiment".format(input_column))

    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the sentiment based on the inputs
    def _get_values(self, inputs):
        analyzer = SentimentIntensityAnalyzer()
        result = []
        for tweet in inputs[0]:
            sentiment = analyzer.polarity_scores(tweet)
            result.append([sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound']])

        return np.array(result)
