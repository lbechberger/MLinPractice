
"""
Created on Thu Oct  18 10:30:41 2021

@author: rfarah
"""

import unittest
import pandas as pd
from nltk import word_tokenize
from code.preprocessing.lemmatizer import Lemmatizer


class LemmatizerTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.lemmatizer = Lemmatizer(self.INPUT_COLUMN, self.OUTPUT_COLUMN)

    def test_input_columns(self):
        self.assertListEqual(self.lemmatizer._input_columns, [self.INPUT_COLUMN])

    def test_output_column(self):
        self.assertEqual(self.lemmatizer._output_column, self.OUTPUT_COLUMN)

    def test_lemmatization(self):
        input_text = "my uncle's cats like playing at our yard where there are many trees."
        input_text_tokenized = [[word_tokenize(input_text)]]
        output_text = "my uncle's cat like play at our yard where there be many tree."
        output_text_tokenized = word_tokenize(output_text)
        input_df = pd.DataFrame(input_text_tokenized, columns=[self.INPUT_COLUMN])
        lemmatized = self.lemmatizer.fit_transform(input_df)
        # print(lemmatized)
        # self.assertListEqual(lemmatized[self.OUTPUT_COLUMN][0], output_text_tokenized)

if __name__ == '__main__':
    unittest.main()


