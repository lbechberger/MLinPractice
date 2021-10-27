#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  14 14:51:00 2021
"""

import unittest
import pandas as pd
import numpy as np
import pdb
from code.feature_extraction.hash_vector import HashVector


class HashVectorTest(unittest.TestCase):
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.hash_vector_feature = HashVector(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = [
            "This is a tweet This is also a test",
            "This is a tweet This is also a test",
            "hallo ne data science",
            "OMG look at this",
        ]
        self.result = self.hash_vector_feature._get_values([self.df.squeeze()])

    def test_input_columns(self):
        self.assertEqual(self.hash_vector_feature._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(
            self.hash_vector_feature.get_feature_name(),
            self.INPUT_COLUMN + "_hashvector",
        )

    def test_result_shape(self):
        self.assertEqual(self.result.shape[0], len(self.df[self.INPUT_COLUMN]))

    def test_result_type(self):
        self.assertEqual(type(self.result), np.ndarray)


if __name__ == "__main__":
    unittest.main()
