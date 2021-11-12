#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for the viral keyword feature extracture

Created: 12.11.21, 15:45

Author: LDankert
"""

import unittest
from src.feature_extraction.keyword import Keyword


class KeywordTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input_column"
        self.LABEL_COLUMN = "label_column"
        self.number_of_words = 3
        self.inputs = [["test", "test"], ["doof","test"], ["ml","doof"]]
        self.expected_variables = [("test", 3), ("doof", 2), ("ml",1)]
        self.expected_outputs = [[1,0,0], [1,1,0], [0,1,1]]
        self.keyword =Keyword([self.INPUT_COLUMN, self.LABEL_COLUMN], self.number_of_keywords)
        self.keyword._set_variables(self.inputs)

    def test_keyword_set_variables(self):
        test_value = [self.expected_variables == self.keyword.keywords]
        self.assertTrue(test_value)

    def test_keywords_get_values(self):
        function_output = self.keywords._get_values(self.inputs)
        test_value = [self.expected_outputs == function_output]
        self.assertTrue(test_value)
    
    def test_keywords_set_variables(self)

    def test_input_columns(self):
        self.assertEqual(self.keywords._input_columns, [self.INPUT_COLUMN, self.LABEL_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.keywords.get_feature_name(), "keywords")


if __name__ == '__main__':
    unittest.main()
