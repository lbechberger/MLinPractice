#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:51:00 2021

@author: ml
"""

import unittest
import pandas as pd
import numpy as np
from code.feature_extraction.likes_counter import LikesCounter


class LikesCounterTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "likes_count"
        self.images = LikesCounter(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = [1, 1, 1, 0, 0, 1, 0, 1, 1, 0]

    def test_input_columns(self):
        self.assertEqual(self.images._input_columns,
                         [self.INPUT_COLUMN])

    def test_number_of_likes(self):
        num_likes = self.images.transform(self.df)
        EXPECTED_NUM_LIKES = [[6]]

        self.assertEqual(np.sum(num_likes), EXPECTED_NUM_LIKES)


if __name__ == '__main__':
    unittest.main()
