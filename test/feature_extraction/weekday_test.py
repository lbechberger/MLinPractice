#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weekday_test

Created: 04.11.21, 12:05

Author: LDankert
"""

import unittest
import pandas as pd
import numpy as np
from src.feature_extraction.weekday import Weekday

class WeeldayFeatureTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "weekday"
        self.inputs = pd.DataFrame(["1994-01-23","1994-01-24","2023-10-31","2021-02-26","2019-09-18","2018-10-20","2021-03-18"])
        self.expected_outputs = [[0,0,0,0,0,0,1], [1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,0,0,1,0,0],
                            [0,0,1,0,0,0,0],[0,0,0,0,0,1,0],[0,0,0,1,0,0,0]]
        self.weekdayer = Weekday(self.INPUT_COLUMN)

    def test_weekday_get_values(self):
        function_output = self.weekdayer._get_values(self.inputs)
        test_value = [self.expected_outputs == function_output]
        self.assertTrue(test_value)

    def test_input_columns(self):
        self.assertEqual(self.weekdayer._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.weekdayer.get_feature_name(), self.INPUT_COLUMN)

if __name__ == '__main__':
    unittest.main()
