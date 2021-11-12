#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for the most common words feature extracture

Created: 12.11.21, 15:34

Author: LDankert
"""

import unittest
from src.feature_extraction.most_common_words import MostCommonWords


class MostCommonWordsTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input_column"
        self.number_of_words = 3
        self.inputs = [["test", "test"], ["doof","test"], ["ml","doof"]]
        self.expected_variables = [("test", 3), ("doof", 2), ("ml",1)]
        self.expected_outputs = [[1,0,0], [1,1,0], [0,1,1]]
        self.most_common_word = MostCommonWords(self.INPUT_COLUMN, self.number_of_words)
        self.most_common_word._set_variables(self.inputs)

    def test_most_common_words_set_variables(self):
        test_value = [self.expected_variables == self.most_common_word.common_words]
        self.assertTrue(test_value)

    def test_most_common_words_get_values(self):
        function_output = self.most_common_word._get_values(self.inputs)
        test_value = [self.expected_outputs == function_output]
        self.assertTrue(test_value)

    def test_input_columns(self):
        self.assertEqual(self.most_common_word._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.most_common_word.get_feature_name(), "most_common_words")


if __name__ == '__main__':
    unittest.main()
