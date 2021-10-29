
"""
Created on Thu Oct  18 10:30:41 2021

@author: rfarah
"""

import unittest
import pandas as pd
from nltk import word_tokenize
from code.preprocessing.stemmer import Stemmer


class StemmmerTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.stemmer = Stemmer(self.INPUT_COLUMN, self.OUTPUT_COLUMN)

    def test_input_columns(self):
        self.assertListEqual(self.stemmer._input_columns, [self.INPUT_COLUMN])

    def test_output_column(self):
        self.assertEqual(self.stemmer._output_column, self.OUTPUT_COLUMN)

    def test_stemming(self):
        input_text = "my uncle's cats like playing at our yard where there are many trees."
        input_text_tokenized = [word_tokenize(input_text)]
        ser = pd.Series(data=[input_text_tokenized, "en"], index=["tweet", "language"])
        output_text = "my uncl's cat like play at our yard where there are mani tree."
        output_text_tokenized = word_tokenize(output_text)
        output_text_tokenized
        input_df = pd.DataFrame(data = [input_text_tokenized, "en"], columns=[self.INPUT_COLUMN])
        stemmed = self.stemmer.fit_transform(input_df)
        self.assertListEqual(stemmed[self.OUTPUT_COLUMN][0], output_text_tokenized)

if __name__ == '__main__':
    unittest.main()


