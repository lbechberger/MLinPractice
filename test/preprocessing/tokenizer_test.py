#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:30:41 2021

@author: ml
"""

import unittest
import pandas as pd
from code.preprocessing.tokenizer import Tokenizer

class TokenizerTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.tokenizer = Tokenizer(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
    
    def test_boolean(self):
        self.assertEqual(True, not False)
    
    def test_input_columns(self):
        self.assertListEqual(self.tokenizer._input_columns, [self.INPUT_COLUMN])

    def test_output_column(self):
        self.assertEqual(self.tokenizer._output_column, self.OUTPUT_COLUMN)

    def test_tokenization_single_sentence(self):
        input_text = "This is an example sentence"
        output_text = "['This', 'is', 'an', 'example', 'sentence']"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        tokenized = self.tokenizer.fit_transform(input_df)
        self.assertEqual(tokenized[self.OUTPUT_COLUMN][0], output_text)
    

if __name__ == '__main__':
    unittest.main()