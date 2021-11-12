#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Most Common Word class for feature extractor

Created: 12.11.21, 12:42

Author: LDankert
"""


import numpy as np
from nltk import FreqDist
from sklearn.preprocessing import MultiLabelBinarizer
from src.feature_extraction.feature_extractor import FeatureExtractor


# class for extracting the most common words in all tweets
class MostCommonWords(FeatureExtractor):

    # constructor
    def __init__(self, input_column, number_of_words):
        super().__init__([input_column], "most_common_words")
        self.number_of_words = number_of_words

    # set the variables depending on the number of words
    def _set_variables(self, inputs):
        all_tweets = []
        for tweet in inputs[0]:
            all_tweets.extend(tweet)

        freq = FreqDist(all_tweets)
        self.common_words = freq.most_common(self.number_of_words)
        print(f"    The {self.number_of_words} most common words:")
        for word in self.common_words:
            print(f"        {word[0]}: {word[1]}")

    # returns columns, one for each most common words
    def _get_values(self, inputs):
        result = []
        common_words = [word for word, _ in self.common_words]
        for tweet in inputs[0]:
            words_in_tweet = []
            words_in_tweet += [word for word in common_words if word in tweet]
            result.append(words_in_tweet)

        enc = MultiLabelBinarizer()
        result = enc.fit_transform(result)
        return result
