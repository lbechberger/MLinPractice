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

    def test_input_columns(self):
        self.assertEqual(self.tokenizer._input_columns, [
                             self.INPUT_COLUMN])

    def test_output_column(self):
        self.assertEqual(self.tokenizer._output_column, self.OUTPUT_COLUMN)

    def test_tokenization_single_sentence(self):
        input_text = "This is an example sentence"
        output_text = ['this', 'is', 'an', 'example', 'sentence']

        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]

        tokenized = self.tokenizer.fit_transform(input_df)
        self.assertEqual(tokenized[self.OUTPUT_COLUMN][0], output_text)

    def test_tokenization_several_sentences(self):
        input_text = ["This is an example sentence",
                      "I want for my tweet to go viral", "Can we talk about global warming?"]
        output_text = [['this', 'is', 'an', 'example', 'sentence'], ["i", "want", "for", "my", "tweet", "to", "go", "viral"],
                       ["can", "we", 'talk', 'about', 'global', 'warming']]

        input_df = pd.DataFrame(input_text, columns=[self.INPUT_COLUMN])

        tokenized = self.tokenizer.fit_transform(input_df)
        for i in range(3):
            self.assertEqual(tokenized[self.OUTPUT_COLUMN][i], output_text[i])

    def test_tokenization_several_sentences_with_urls_hashtags_mentions_numberes(self):
        input_text = ["This is an example sentence (7/7) https://www.atlassian.com/git/tutorials/using-branches",
                      "I want for my tweet to go viral 100% @twitter", "Can we talk about global warming? #hashtag"]
        output_text = [['this', 'is', 'an', 'example', 'sentence'], ["i", "want", "for", "my", "tweet", "to", "go", "viral"],
                       ["can", "we", 'talk', 'about', 'global', 'warming']]

        input_df = pd.DataFrame(input_text, columns=[self.INPUT_COLUMN])

        tokenized = self.tokenizer.fit_transform(input_df)
        for i in range(3):
            self.assertEqual(tokenized[self.OUTPUT_COLUMN][i], output_text[i])


if __name__ == '__main__':
    unittest.main()
