#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:51:00 2021

@author: ml
"""

import unittest
import pandas as pd
from code.feature_extraction.similar_tweets import SimilarTweets


class SimilarTweetsTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "stemmed_tweet"
        self.sentiment_analyser = SimilarTweets(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ["['scot', 'show', 'world', 'excel', 'applianc', 'data', 'scienc']", "['machinelearn', 'fun', 'mathemat', 'learn', 'experi']", "['hate', 'the', 'new', 'sophisticated', 'layout']", "['case', 'per', 'student', 'daili', 'collegian', 'data', 'analysi', 'found']",
                                      "['umass', 'covid-19', 'respons', 'stack', 'school', 'massachusett', 'univers', 'rank', 'third', 'highest', 'state', 'posit', 'case', 'per', 'student', 'daili', 'collegian', 'data', 'analysi', 'found']"]

    def test_input_columns(self):
        self.assertEqual(self.sentiment_analyser._input_columns,
                         [self.INPUT_COLUMN])

    def test_sentiment_analysis(self):
        results = self.sentiment_analyser.transform(self.df)
        # check the input and the output length
        EXPECTED_LENGTH = 25
        self.assertEqual(results.shape[0], EXPECTED_LENGTH)
        # the third one and the last one should have high cosine similarity value
        EXPECTED_SIMILARITY_THRESHOLD = 0.8
        self.assertTrue(results[24] > EXPECTED_SIMILARITY_THRESHOLD)


if __name__ == '__main__':
    unittest.main()
