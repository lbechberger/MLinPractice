#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lowercaser_test

Created: 11.11.21, 18:09

Author: LDankert
"""

import unittest
import pandas as pd
from src.preprocessing.lowercaser import Lowercaser


class LowercaserTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.lowercaser = Lowercaser(self.INPUT_COLUMN, self.OUTPUT_COLUMN)

    def test_boolean(self):
        self.assertEqual(True, not False)

    def test_input_columns(self):
        self.assertListEqual(self.lowercaser._input_columns, [self.INPUT_COLUMN])

    def test_output_column(self):
        self.assertEqual(self.lowercaser._output_column, self.OUTPUT_COLUMN)

    def test_tokenization_single_sentence(self):
        input_text = "This Is aN exAMPle senTEnce"
        output_text = "this is an example sentence"

        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]

        lowercaser = self.lowercaser.fit_transform(input_df)
        self.assertEqual(lowercaser[self.OUTPUT_COLUMN][0], output_text)


if __name__ == '__main__':
    unittest.main()
