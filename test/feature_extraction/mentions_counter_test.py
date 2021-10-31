#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:51:00 2021

@author: ml
"""

import unittest
import pandas as pd
from code.feature_extraction.mentions_counter import MentionsCounter


class MentionsCounterTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "mentions"
        self.mentions = MentionsCounter(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = [
            "[{'screen_name': 'lockheedmartin', 'name': 'lockheed martin', 'id': '42871498'}, {'screen_name': 'sassoftware', 'name': 'sas software', 'id': '112464786'}]"]

    def test_input_columns(self):
        self.assertEqual(self.mentions._input_columns,
                         [self.INPUT_COLUMN])

    def test_number_of_mentions(self):
        num_mentions = self.mentions.transform(self.df)
        EXPECTED_NUM_MENTIONS = [[2]]
        self.assertEqual(num_mentions, EXPECTED_NUM_MENTIONS)


if __name__ == '__main__':
    unittest.main()
