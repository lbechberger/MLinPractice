#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests feature extraction
"""

import unittest
import pandas as pd
import numpy as np
from src.feature_extraction.counter_fe import CounterFE

class CountFeatureTest(unittest.TestCase):
    
    def setUp(self):       
        self.INPUT_COLUMN = "mockcolumn"
        self.count_feature_extractor = CounterFE(self.INPUT_COLUMN)
        
        self.df = pd.DataFrame()        
        self.df[self.INPUT_COLUMN] = [
            "[{'screen_name': 'zeebusiness', 'name': 'zee business', 'id': '140798905'}, {'screen_name': 'amishdevgan', 'name': 'amish devgan', 'id': '163817624'}]",
            "[]",
            "[{'screen_name': 'zeebusiness', 'name': 'zee business', 'id': '140798905'}]"
        ]


    def test_feature_name(self):
        self.assertEqual(self.count_feature_extractor.get_feature_name(), "mockcolumn_count")


    def test_counting(self):
        
        self.count_feature_extractor.fit(self.df)

        actual_feature = self.count_feature_extractor.transform(self.df)        
        EXPECTED = np.array(pd.DataFrame({"mockcolumn_count": [2, 0, 1]}))

        isEqual = np.array_equal(actual_feature, EXPECTED, equal_nan=False)
        self.assertTrue(isEqual)
        

if __name__ == '__main__':
    unittest.main()