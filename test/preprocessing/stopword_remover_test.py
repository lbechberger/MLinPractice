#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stopword_remover_test

Created: 11.11.21, 18:37

Author: LDankert
"""


import unittest
import pandas as pd
from src.preprocessing.stopword_remover import StopwordRemover


class StopwordRemoverTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.stopwords_remover = StopwordRemover(self.INPUT_COLUMN, self.OUTPUT_COLUMN)

    def test_boolean(self):
        self.assertEqual(True, not False)

    def test_input_columns(self):
        self.assertListEqual(self.stopwords_remover._input_columns, [self.INPUT_COLUMN])

    def test_output_column(self):
        self.assertEqual(self.stopwords_remover._output_column, self.OUTPUT_COLUMN)

    def test_tokenization_single_sentence(self):
        input_text = ['this', 'is', 'an', 'example', 'sentence']
        output_text = ['example', 'sentence']

        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]

        stopwords_remover = self.stopwords_remover.fit_transform(input_df)
        self.assertEqual(stopwords_remover[self.OUTPUT_COLUMN][0], output_text)


if __name__ == '__main__':
    unittest.main()
