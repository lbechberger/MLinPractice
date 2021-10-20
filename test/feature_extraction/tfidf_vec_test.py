#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:51:00 2021

@author: ml
"""

import unittest
import pandas as pd
import numpy as np


from code.feature_extraction.tfidf_vector import TfidfVector


class TfidfVectorTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "preprocessing_col"
        self.tfidf_vector_feature = TfidfVector(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ["This is a tweet This is also a test",
                                      "This is a tweet This is also a test", "hallo ne data science", "OMG look at this"]

        self.result = self.tfidf_vector_feature._get_values([self.df.squeeze()])

    def test_input_columns(self):
        self.assertEqual(self.tfidf_vector_feature._input_columns, [
                         self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.tfidf_vector_feature.get_feature_name(
        ), self.INPUT_COLUMN + "_tfidfvector")

    def test_result_shape(self):
        self.assertEqual(self.result.shape[0], len(self.df[self.INPUT_COLUMN]))

    def test_result_positive(self):
        self.assertTrue(np.all(self.result >= 0))


if __name__ == '__main__':
    unittest.main()
