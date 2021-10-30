#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:51:00 2021

@author: ml
"""

import unittest
from nltk.util import pr
import pandas as pd
from code.feature_extraction.images_counter import Images


class ImageTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "photos"
        self.images = Images(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ["['https://pbs.twimg.com/media/D5afNaGX4AA3gC9.jpg', 'https://pbs.twimg.com/media/D5afNaKXkAAZbpo.jpg', 'https://pbs.twimg.com/media/D5afNaKWsAIab9W.jpg']"]

    def test_input_columns(self):
        self.assertEqual(self.images._input_columns,
                         [self.INPUT_COLUMN])

    def test_number_of_photos(self):
        num_photos = self.images.transform(self.df)
        EXPECTED_NUM_PHOTOS = [[3]]
        print(num_photos)
        self.assertEqual(num_photos, EXPECTED_NUM_PHOTOS)


if __name__ == '__main__':
    unittest.main()
