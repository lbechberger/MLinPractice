#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for the stemmer class

Created: 11.11.21, 19:11

Author: LDankert
"""


import unittest
import pandas as pd
from src.preprocessing.stemmer import Stemmer


class StemmerTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.stemmer = Stemmer(self.INPUT_COLUMN, self.OUTPUT_COLUMN)

    def test_boolean(self):
        self.assertEqual(True, not False)

    def test_input_columns(self):
        self.assertListEqual(self.stemmer._input_columns, [self.INPUT_COLUMN])

    def test_output_column(self):
        self.assertEqual(self.stemmer._output_column, self.OUTPUT_COLUMN)

    def test_tokenization_single_sentence(self):
        input_text = ["game","gaming","gamed","games"]
        output_text = ["game", "game", "game", "game"]

        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]

        stemmer = self.stemmer.fit_transform(input_df)
        self.assertEqual(stemmer[self.OUTPUT_COLUMN][0], output_text)


if __name__ == '__main__':
    unittest.main()
