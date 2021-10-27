#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfVector(FeatureExtractor):
    """
        TfidfVector to create a number matrix out of text data.
        Read sklearn documentation for more information.
    """

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_tfidfvector".format(input_column))

    # don't need to fit, so don't overwrite _set_variables()

    def _get_values(self, inputs):
        # inputs is list of text documents
        # create the transform object

        vectorizer = TfidfVectorizer(
            strip_accents="ascii", stop_words="english", ngram_range=(1, 2)
        )
        # encode document
        vector = vectorizer.fit_transform(inputs[0])

        return vector.toarray()
