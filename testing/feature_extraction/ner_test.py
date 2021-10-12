#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unittester for the named-entity-recognition feature extractor.
Created on Sat Oct 9 23:59:37 2021
@author: louiskhub
"""

import unittest
import pandas as pd
import numpy as np
from code.feature_extraction.ner import NER
from numpy.testing import assert_array_equal

class NERtest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMNS = "input"
        self.ner = NER(self.INPUT_COLUMNS)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMNS] = [
            "these new data will ultimately help scientists more accurately project the fate of the glacier   how quickly it is melting and retreating inland  and how far it might be from complete collapse   ",
            "in a video by  lenovo   ronald vanloon shares how  innovation comes from data and how people use it  “putting people at the center of decision making becomes a tail wind that helps drive all other measures of success ” the think human report   datascience  ",
            "did you know that  datascience is one of the highest paid technology jobs   join our webinar tomorrow evening to learn about our undergraduate course  register here     cao2021"]

    def test_input_columns(self):
        self.assertEqual(self.ner._input_columns, [self.INPUT_COLUMNS])

    def test_feature_name(self):
        self.assertEqual(self.ner.get_feature_name(), self.INPUT_COLUMNS + "_ner")

    def test_list_of_entities_exists(self):
        output = self.ner.fit_transform(self.df)
        self.assertGreaterEqual(len(list(output)), 0)

    def test_output_is_correct(self):
        EXPECTED_OUTPUT = np.array([
            [2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0.]])
        output = self.ner.fit_transform(self.df)
        assert_array_equal(output,EXPECTED_OUTPUT)

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