#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:51:00 2021

@author: ml
"""

import unittest
import pandas as pd
import nltk
from code.feature_extraction.hash_vector import HashVector

class HashVectorTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.hash_vector_feature = HashVector(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ['["This", "is", "a", "tweet", "This", "is", "also", "a", "test"]', '["This", "is", "a", "tweet", "This", "is", "also", "a", "test"]']
    
    def test_input_columns(self):
        self.assertEqual(self.hash_vector_feature._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.hash_vector_feature.get_feature_name(), self.INPUT_COLUMN + "_hashvector")





if __name__ == '__main__':
    unittest.main()