#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:51:00 2021

@author: ml
"""

import unittest
import pandas as pd
import numpy as np
from code.feature_extraction.sentiment_analysis import SentimentAnalysis


class SentimentAnalysisTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "stemmed_tweet"
        self.sentiment_analyser = SentimentAnalysis(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ["['scot', 'show', 'world', 'excel', 'applianc', 'data', 'scienc']", "['machinelearn', 'fun', 'mathemat', 'learn', 'experi']", "['hate', 'the', 'new', 'sophisticated', 'layout']",
                                      "['umass', 'covid-19', 'respons', 'stack', 'school', 'massachusett', 'univers', 'rank', 'third', 'highest', 'state', 'posit', 'case', 'per', 'student', 'daili', 'collegian', 'data', 'analysi', 'found']"]

    def test_input_columns(self):
        self.assertEqual(self.sentiment_analyser._input_columns,
                         [self.INPUT_COLUMN])

    def test_sentiment_analysis(self):
        results = self.sentiment_analyser.transform(self.df)
        POS = [0., 0., 1.]
        NEG = [1., 0., 0.]
        NEU = [0., 1., 0.]
        EXPECTED_SENTIMENT = [POS, POS, NEG, POS]
        for i in range(3):
            self.assertListEqual(list(results[i][0]), EXPECTED_SENTIMENT[i])


if __name__ == '__main__':
    unittest.main()
