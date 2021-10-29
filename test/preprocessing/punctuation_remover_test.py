"""[summary]
"""
import unittest
import pandas as pd
from nltk import word_tokenize
from code.preprocessing.punctuation_remover import PunctuationRemover

class PunctuationRemoverTest(unittest.TestCase):
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.punctuation_remover = PunctuationRemover(self.INPUT_COLUMN, self.OUTPUT_COLUMN)

    def test_input_columns(self):
        self.assertListEqual(self.punctuation_remover._input_columns, [self.INPUT_COLUMN])

    def test_output_column(self):
        self.assertEqual(self.punctuation_remover._output_column, self.OUTPUT_COLUMN)

    def test_punctuation_remover(self):
        input_text = "it's a sentence with punctuation, #hashtags, @mentions, question marks? :) >.< *.*'"
        input_text_tokenized = [[word_tokenize(input_text)]]
        output_text = "it's a sentence with punctuation hashtags mentions question marks"
        output_text_tokenized = word_tokenize(output_text)
        input_df = pd.DataFrame(input_text_tokenized, columns=[self.INPUT_COLUMN])
        punctuationremoved = self.punctuation_remover.fit_transform(input_df)
        self.assertListEqual(punctuationremoved[self.OUTPUT_COLUMN][0], output_text_tokenized)


if __name__ == '__main__':
    unittest.main()