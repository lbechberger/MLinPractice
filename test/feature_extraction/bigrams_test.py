#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:51:00 2021

@author: ml
"""

import unittest
import pandas as pd
import nltk
from code.feature_extraction.bigrams import BigramFeature

class BigramFeatureTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.bigram_feature = BigramFeature(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ['["This", "is", "a", "tweet", "This", "is", "also", "a", "test"]']
    
    def test_input_columns(self):
        self.assertEqual(self.bigram_feature._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.bigram_feature.get_feature_name(), self.INPUT_COLUMN + "_bigrams")

    def test_list_of_bigrams_exists(self):
        self.bigram_feature.fit(self.df)
        self.assertGreaterEqual(len(list(self.bigram_feature._bigrams)), 0)

    def test_list_of_bigrams_most_frequent_correct(self):
        self.bigram_feature.fit(self.df)
        EXPECTED_BIGRAM = ('This', 'is')
        
        freq_dist = nltk.FreqDist(self.bigram_feature._bigrams)
        freq_list = list(freq_dist.items())
        freq_list.sort(key = lambda x: x[1], reverse = True)
        
        self.assertEqual(freq_list[0][0], EXPECTED_BIGRAM)

if __name__ == '__main__':
    unittest.main()