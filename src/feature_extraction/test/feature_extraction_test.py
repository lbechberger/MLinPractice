#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests feature extraction
"""

import csv
import logging
import unittest
import pandas as pd
import numpy as np
from src.feature_extraction.counter_fe import CounterFE

class CountFeatureTest(unittest.TestCase):
    
    def setUp(self):
        logging.basicConfig()
        self.log = logging.getLogger("LOG")

        self.tryout_df = pd.read_csv("data/preprocessing/split/training.csv", quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")
        self.tryout_df = self.tryout_df.rename(columns={"mentions": "mockcolumn", "photos": "mockphotos"})        

        self.INPUT_COLUMN = "mockcolumn"
        self.counter_feature = CounterFE(self.INPUT_COLUMN)
        # self.df = pd.DataFrame({ self.INPUT_COLUMN: [{'screen_name': 'zeebusiness', 'name': 'zee business', 'id': '140798905'}, {'screen_name': 'amishdevgan', 'name': 'amish devgan', 'id': '163817624'}] } )

        self.df = pd.DataFrame()
        # self.df[self.INPUT_COLUMN] = "['[\"This\", \"row\", \"has\", \"five\", \"elements\"], [\"this\", \"only\", \"thre\"], [\"one\"], []']"
        self.df[self.INPUT_COLUMN] = [
            "[{'screen_name': 'zeebusiness', 'name': 'zee business', 'id': '140798905'}, {'screen_name': 'amishdevgan', 'name': 'amish devgan', 'id': '163817624'}]",
            "[]",
            "[{'screen_name': 'zeebusiness', 'name': 'zee business', 'id': '140798905'}]"
            ]
        print("")


    def test_input_columns(self):
        self.assertEqual(self.counter_feature._input_columns, [self.INPUT_COLUMN])


    def test_feature_name(self):
        self.assertEqual(self.counter_feature.get_feature_name(), self.INPUT_COLUMN + "_count")


    def test_counting(self):
        self.counter_feature.fit(self.df)
        actual_feature = self.counter_feature.transform(self.df)
        # actual_feature = self.counter_feature.transform(self.tryout_df)
        # EXPECTED = np.array(pd.DataFrame({"mockcolumn_count": [5,3,1,0]}))
        EXPECTED = np.array(pd.DataFrame({"mockcolumn_count": [2,0,1]}))

        # self.log.warning("actual_feature", actual_feature)
        # self.log.warning("EXPECTED", EXPECTED)

        isEqual = np.array_equal(actual_feature, EXPECTED, equal_nan=False)
        self.assertTrue(isEqual)
        

if __name__ == '__main__':
    unittest.main()