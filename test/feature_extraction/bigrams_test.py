#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:51:00 2021

@author: ml
"""

import unittest
import pandas as pd
from code.feature_extraction.bigrams import BigramFeature


class BigramFeatureTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.bigram_feature = BigramFeature(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ['["This", "is", "a", "tweet", "This", "is", "also", "a", "test"]',
        '["This", "is", "a", "lovely" ,"cat", "very", "lovely", "cat"]']

    
    def test_input_columns(self):
        self.assertEqual(self.bigram_feature._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.bigram_feature.get_feature_name(), self.INPUT_COLUMN + "_bigrams")

    def test_list_of_bigrams_exists(self):
        self.bigram_feature.fit(self.df)
        self.assertGreaterEqual(len(list(self.bigram_feature._bigrams_of_all_tweets)), 0)


    def test_input_columns(self):
        self.assertEqual(self.bigram_feature._input_columns,
                         [self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.bigram_feature.get_feature_name(),
                         self.INPUT_COLUMN + "_bigrams")

    def test_list_of_bigrams_exists(self):
        self.bigram_feature.transform(self.df)
        self.assertGreaterEqual(len(list(self.bigram_feature._bigrams_of_all_tweets)), 0)


    def test_list_of_bigrams_most_frequent_correct(self):
        self.bigram_feature.transform(self.df)
        EXPECTED_BIGRAM = [('This', 'is'), 3, ('is', 'a'), 2, ('lovely', 'cat'), 2]
        freq_dist = self.bigram_feature._freq_list_testing
        for i in range(2):
            j = 0
            self.assertEqual(freq_dist[0][j][i], EXPECTED_BIGRAM[i])
            j += 1


if __name__ == '__main__':
    unittest.main()
    
