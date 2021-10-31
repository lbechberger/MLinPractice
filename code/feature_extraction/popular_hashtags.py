#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find the frequently used set of hashtags by finding their tfId values and sum them.

Created on Wed Sep 29 12:29:25 2021

@author: lbechberger
"""
import ast
import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor
from sklearn.feature_extraction.text import TfidfVectorizer


class PopularHashtags(FeatureExtractor):

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column],
                         "{0}_frequently_used".format(input_column))
        # for testing
        self.most_popular_tweet = 0

    def _get_values(self, inputs):
        concatenated_hashtags = [
            " ".join(ast.literal_eval(x)) for x in inputs[0]]
        tf_id = TfidfVectorizer()
        vecs = tf_id.fit_transform(concatenated_hashtags).todense()

        result = np.array(np.sum(vecs, axis=1))

        self.most_popular_tweet = np.unravel_index(
            np.argmax(result, axis=None), result.shape)
        result = result.reshape(-1, 1)
        return result
