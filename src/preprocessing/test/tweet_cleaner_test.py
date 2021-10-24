#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:30:41 2021

@author: ml
"""

import unittest
import pandas as pd
from src.preprocessing.tweet_clean import TweetClean
from src.util import fm

class TweetCleanerTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "some_column"
        self.OUTPUT_COLUMN = "some_column_cleaned"
        self.cleaner = TweetClean(self.INPUT_COLUMN, self.OUTPUT_COLUMN)

    def _apply_transform(self, input_sentence):
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_sentence]
        
        cleaned = self.cleaner.transform(input_df)
        cleaned_column = cleaned[self.OUTPUT_COLUMN][0]
        return cleaned_column

    def test_punctuation_removal(self):
        input_sentence = "This is an example sentence. And another sentence . And more"
        expected_output_text = "This is an example sentence And another sentence  And more"
        
        msg = fm("sentences with urls", "returns sentences without urls")
        self.assertEqual(self._apply_transform(input_sentence), expected_output_text, msg)     

    def test_url_removal1(self):
        input_sentence = "This url will be removed https://example.org hopefully!"
        expected_output_text = "This url will be removed hopefully!"
        
        msg = fm("sentences with urls", "returns sentence without urls")
        self.assertEqual(self._apply_transform(input_sentence), expected_output_text, msg)     
    
    def test_url_removal2(self):
        input_sentence = "http://t.co/DOFVEUCiBV Big Data needs data science but data science doesn't need big data"
        expected_output_text = "Big Data needs data science but data science doesn't need big data"
        
        msg = fm("sentences with urls", "returns sentence without urls")
        self.assertEqual(self._apply_transform(input_sentence), expected_output_text, msg)     

    def test_cleaning(self):
        input_sentence = "#DataScience is greater than the sum of its parts https://t.co/lMcc9OJwWr #BigData #Analytics | RT @Ronald_vanLoon  https://t.co/UT8RFLoAy4"
        expected_output_text = "DataScience is greater than the sum of its parts BigData Analytics RT Ronald_vanLoon  "
        
        msg = fm("sentences with urls and hashtags", "returns cleaned sentence")
        self.assertEqual(self._apply_transform(input_sentence), expected_output_text, msg)     


if __name__ == '__main__':
    unittest.main()