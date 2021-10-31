#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:51:00 2021

@author: ml
"""

import unittest
import pandas as pd
from code.feature_extraction.replies_counter import RepliesCounter


class RepliesCounterTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "replies_scount"
        self.counter = RepliesCounter(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = [0, 1, 4, 5, 6, 7]

    def test_input_columns(self):
        self.assertEqual(self.counter._input_columns,
                         [self.INPUT_COLUMN])

    def test_number_of_replies(self):
        num_replies = self.counter.transform(self.df)

        EXPECTED_NUM = [[0], [1], [4], [5], [6], [7]]
        for i in range(6):
            self.assertEqual(num_replies[i], EXPECTED_NUM[i])


if __name__ == '__main__':
    unittest.main()
