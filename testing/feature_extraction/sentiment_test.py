#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
import unittest
import pandas as pd
from code.feature_extraction.sentiment import Sentiment


class SentimentTest(unittest.TestCase):

    def setUp(self):
        self._sentiment_extractor = Sentiment('tweet')

    def test_fit_transform_positive(self):
        df = pd.DataFrame()
        df['tweet'] = ['This is a very nice positively good tweet.']
        output = self._sentiment_extractor.fit_transform(df)
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 4)

        self.assertEqual(output[0, 0], 0.0)  # negative
        self.assertGreater(output[0, 1], 0.5)  # positive
        self.assertLess(output[0, 2], 0.5)  # neutral
        self.assertGreater(output[0, 3], 0)  # compound

    def test_fit_transform_negative(self):
        df = pd.DataFrame()
        df['tweet'] = ['This is a very nasty negatively bad tweet.']
        output = self._sentiment_extractor.fit_transform(df)
        self.assertGreater(output[0, 0], 0.5)  # negative
        self.assertEqual(output[0, 1], 0.0)  # positive
        self.assertLess(output[0, 2], 0.5)  # neutral
        self.assertLess(output[0, 3], 0)  # compound
