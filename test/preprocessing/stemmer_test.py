
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
        self.INPUT_COLUMN = ["input", "language"]
        self.OUTPUT_COLUMN = "output"
        self.stemmer = Stemmer(self.INPUT_COLUMN, self.OUTPUT_COLUMN)

    def test_input_columns(self):
        self.assertListEqual(self.stemmer._input_columns, [self.INPUT_COLUMN])

    def test_output_column(self):
        self.assertEqual(self.stemmer._output_column, self.OUTPUT_COLUMN)

    def test_english_stemmer(self):
        input_text = "my uncle's cats like playing at our yard where there are many trees."
        input_text_tokenized = [[word_tokenize(input_text)]]
        output_text = "my uncl's cat like play at our yard where there are mani tree."
        output_text_tokenized = word_tokenize(output_text)
        input_df = pd.DataFrame(
            {self.INPUT_COLUMN[0]: input_text_tokenized, self.INPUT_COLUMN[1]: 'en'})
        stemmed = self.stemmer.fit_transform(input_df)
        self.assertListEqual(
            stemmed[self.OUTPUT_COLUMN][0], output_text_tokenized)

    def test_german_stemmer(self):
        input_text = "wir müssen das interessante Projekt fertig machen bis zum Ende dieser Woche!"
        input_text_tokenized = [[word_tokenize(input_text)]]
        output_text = "wir muss das interessant Projekt fertig mach bis zum End dies Woch!".lower()
        output_text_tokenized = word_tokenize(output_text)
        input_df = pd.DataFrame(
            {self.INPUT_COLUMN[0]: input_text_tokenized, self.INPUT_COLUMN[1]: 'de'})
        stemmed = self.stemmer.fit_transform(input_df)
        self.assertListEqual(
            stemmed[self.OUTPUT_COLUMN][0], output_text_tokenized)

    def test_arabic_stemmer(self):
        input_text = "مكتبة لمعالجة الكلمات العربية وتجذيعها"
        input_text_tokenized = [[word_tokenize(input_text)]]
        output_text = "مكتب معالج كلم عرب تجذيع"
        output_text_tokenized = word_tokenize(output_text)
        input_df = pd.DataFrame(
            {self.INPUT_COLUMN[0]: input_text_tokenized, self.INPUT_COLUMN[1]: 'ar'})
        stemmed = self.stemmer.fit_transform(input_df)
        self.assertListEqual(
            stemmed[self.OUTPUT_COLUMN][0], output_text_tokenized)

    def test_french_stemmer(self):
        input_text = "Pour ces deux causes, à cette époque de la vie si gaie pour les autres enfants, mais pas pour moi :(."
        input_text_tokenized = [[word_tokenize(input_text)]]
        output_text = "Pour ce deux caus, à cet époqu de la vi si gai pour le autr enfant, mais pas pour moi :(.".lower(
        )
        output_text_tokenized = word_tokenize(output_text)
        input_df = pd.DataFrame(
            {self.INPUT_COLUMN[0]: input_text_tokenized, self.INPUT_COLUMN[1]: 'fr'})
        stemmed = self.stemmer.fit_transform(input_df)
        self.assertListEqual(
            stemmed[self.OUTPUT_COLUMN][0], output_text_tokenized)


if __name__ == '__main__':
    unittest.main()
