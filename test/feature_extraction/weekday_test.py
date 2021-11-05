#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weekday_test

Created: 04.11.21, 12:05

Author: LDankert
"""

import unittest
import pandas as pd
from src.feature_extraction.weekday import Weekday

class WeeldayFeatureTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "dummy name"
        self.inputs = [["Sunday","Monday","Tuesday","Friday","Wednesday","Saturday","Thursday"]]
        self.error_inputs =[["Shitday"],["asda s"], [10], ["Sunday","Monday","Shitday"]]
        self.expected_outputs = [[0,0,0,0,0,0,1], [1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,0,0,1,0,0],
                            [0,0,1,0,0,0,0],[0,0,0,0,0,1,0],[0,0,0,1,0,0,0]]
        self.weekdayer = Weekday(self.INPUT_COLUMN)

    def test_weekday_get_values(self):
        function_output = self.weekdayer._get_values(self.inputs)
        test_value = [self.expected_outputs == function_output]
        self.assertTrue(test_value)

    def test_weekday_get_values_exception(self):
        for input in self.error_inputs:
            self.assertRaises(Exception, self.weekdayer._get_values, input)

    def test_input_columns(self):
        self.assertEqual(self.weekdayer._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.weekdayer.get_feature_name(), self.INPUT_COLUMN)

if __name__ == '__main__':
    unittest.main()
