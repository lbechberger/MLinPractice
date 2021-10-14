"""[summary]
"""

import unittest
import pandas as pd
from code.preprocessing.stopwords_remover import StopwordsRemover

class StopwordsRemoverTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = ["input","language"]
        self.OUTPUT_COLUMN = "output"
        self.stopwordsremover = StopwordsRemover(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
    
    def test_input_columns(self):
        self.assertListEqual(self.stopwordsremover._input_columns, [self.INPUT_COLUMN])

    def test_output_column(self):
        self.assertEqual(self.stopwordsremover._output_column, self.OUTPUT_COLUMN)

    def test_removing_english_stopwords(self):
        input_text = "this is a sentence that shows an example of a text containing some English stopwords."
        input_text_tokenized = [[input_text.split()]]
        output_text = "sentence shows example text containing English stopwords."
        output_text_tokenized = output_text.split()
        input_df = pd.DataFrame({self.INPUT_COLUMN[0]: input_text_tokenized, self.INPUT_COLUMN[1]:'en'})
        stopwordsremoved = self.stopwordsremover.fit_transform(input_df)
        self.assertListEqual(stopwordsremoved[self.OUTPUT_COLUMN][0], output_text_tokenized)

    def test_removing_french_stopwords(self):
        input_text = "qu'est-ce que vous aimez faire pendant votre temps libre?"
        input_text_tokenized = [[input_text.split()]]
        output_text = "qu'est-ce aimez faire pendant temps libre?"
        output_text_tokenized = output_text.split()
        input_df = pd.DataFrame({self.INPUT_COLUMN[0]: input_text_tokenized, self.INPUT_COLUMN[1]:'fr'})
        stopwordsremoved = self.stopwordsremover.fit_transform(input_df)
        self.assertListEqual(stopwordsremoved[self.OUTPUT_COLUMN][0], output_text_tokenized)
        
    def test_removing_german_stopwords(self):
        input_text = "wir müssen das Projekt fertig machen bis Ende dieser Woche!"
        input_text_tokenized = [[input_text.split()]]
        output_text = "müssen Projekt fertig Ende Woche!"
        output_text_tokenized = output_text.split()
        input_df = pd.DataFrame({self.INPUT_COLUMN[0]: input_text_tokenized, self.INPUT_COLUMN[1]:'de'})
        stopwordsremoved = self.stopwordsremover.fit_transform(input_df)
        self.assertListEqual(stopwordsremoved[self.OUTPUT_COLUMN][0], output_text_tokenized)

    def test_removing_contractions(self):
        input_text = "i'd've been working out this week but I was busy, let's see if i'll work out at my parents' place."
        input_text_tokenized = [[input_text.split()]]
        output_text = "would working week busy, let us see work parents' place."
        output_text_tokenized = output_text.split()
        input_df = pd.DataFrame({self.INPUT_COLUMN[0]: input_text_tokenized, self.INPUT_COLUMN[1]:'en'})
        stopwordsremoved = self.stopwordsremover.fit_transform(input_df)
        self.assertListEqual(stopwordsremoved[self.OUTPUT_COLUMN][0], output_text_tokenized)

if __name__ == '__main__':
    unittest.main()  