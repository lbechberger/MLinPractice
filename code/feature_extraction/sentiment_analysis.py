#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for analysing the sentiment of the tweets.Inspired by this nice article:
https://towardsdatascience.com/the-best-python-sentiment-analysis-package-1-huge-common-mistake-d6da9ad6cdeb

Created on Wed Sep 29 12:29:25 2021

@author: rfarah
"""
import ast
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from code.feature_extraction.feature_extractor import FeatureExtractor
# from flair.models import TextClassifier
# from flair.data import Sentence
from nltk.sentiment import SentimentIntensityAnalyzer


class SentimentAnalysis(FeatureExtractor):

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_sentiment".format(input_column))

    def _get_values(self, inputs):
        # self._sia = TextClassifier.load('en-sentiment')
        self._sia = SentimentIntensityAnalyzer()
        self._set_variables(inputs)
        features = np.array([["positive"], ["neutral"], ["negative"]])
        self._encoder = OneHotEncoder(sparse=False)
        self._onehot_encoding = self._encoder.fit_transform(features)

        result = np.array(inputs[0].apply(self._predict))
        result = result.reshape(-1, 1)
        return result

    def _predict(self, input):
        input = ast.literal_eval(input)
        concatenated_tokens = " ".join(input)
        # sentence = Sentence(concatenated_tokens)
        # self._sia.predict(sentence)
        # score = sentence.labels[0]

        sentiment_score = self._sia.polarity_scores(
            concatenated_tokens)['compound']
        # if "POSITIVE" in str(score):
        if sentiment_score > 0:
            return list(self._onehot_encoding[0])  # for positive [0., 0., 1.]
        # elif "NEGATIVE" in str(score):
        elif sentiment_score < 0:
            return list(self._onehot_encoding[2])
        else:
            return list(self._onehot_encoding[1])
