#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unittester for the named-entity-recognition feature extractor.
Created on Sat Oct 9 23:59:37 2021
@author: louiskhub
"""

import unittest
import pandas as pd
from code.feature_extraction.ner import NER

class NERtest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.ner = NER(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ["['data', 'hand', 'researcher', 'confidently', 'say', 'variant', 'b', '__NUMBER__', '__NUMBER__', '__NUMBER__', 'transmissible', 'original', 'covid', '__NUMBER__', 'strain', 'deadly', '\U0001f9a0👨', '🔬']"]

    def test_input_columns(self):
        self.assertEqual(self.ner._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.ner.get_feature_name(), self.INPUT_COLUMN + "_ner")

    def test_list_of_entities_exists(self):
        self.ner.fit(self.df)
        self.assertGreaterEqual(len(list(self.ner)), 0)

if __name__ == '__main__':
    unittest.main() 




"""def setUp(self):
        self._sentiment_extractor = Sentiment('tweet')

def test_fit_transform_positive(self):
    df = pd.DataFrame()
    df['tweet'] = ['This is a very nice positively good tweet.']
    output = self._sentiment_extractor.fit_transform(df)
    self.assertEqual(output.shape[0], 1)
    self.assertEqual(output.shape[1], 4)

    self.assertEqual(output[0, 0], 0.0)  # negative
    self.assertGreater(output[0, 1], 0.5)  # positive
    self.assertLess(output[0, 2], 0.5)  # neutral
    self.assertGreater(output[0, 3], 0)  # compound

def test_fit_transform_negative(self):
    df = pd.DataFrame()
    df['tweet'] = ['This is a very nasty negatively bad tweet.']
    output = self._sentiment_extractor.fit_transform(df)
    self.assertGreater(output[0, 0], 0.5)  # negative
    self.assertEqual(output[0, 1], 0.0)  # positive
    self.assertLess(output[0, 2], 0.5)  # neutral
    self.assertLess(output[0, 3], 0)  # compound"""