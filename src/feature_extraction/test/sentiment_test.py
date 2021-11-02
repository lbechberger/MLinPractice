#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests feature extraction
"""

import unittest
import pandas as pd
import logging
import numpy as np
from src.feature_extraction.sentiment_fe import SentimentFE


class CountFeatureTest(unittest.TestCase):

    def setUp(self):
        logging.basicConfig()
        self.log = logging.getLogger("LOG")

        self.INPUT_COLUMN = "mockcolumn"
        self.sentimentFE = SentimentFE(self.INPUT_COLUMN)

        self.df = pd.DataFrame()

        # example taken from official code examples
        # https://github.com/cjhutto/vaderSentiment#code-examples
        self.df[self.INPUT_COLUMN] = [
            # positive sentence example
            "VADER is smart, handsome, and funny.",
            # punctuation emphasis handled correctly (sentiment intensity adjusted)
            "VADER is smart, handsome, and funny!",
            # booster words handled correctly (sentiment intensity adjusted)
            "VADER is very smart, handsome, and funny.",
        ]

    def test_feature_name(self):
        self.assertEqual(self.sentimentFE.get_feature_name(),
                         "mockcolumn_sentiment")

    def test_counting(self):

        self.sentimentFE.fit(self.df)

        actual_feature = self.sentimentFE.transform(self.df)
        # actual_feature = actual_feature.to_numpy()

        expected_values = [
            {'pos': 0.746, 'compound': 0.8316, 'neu': 0.254, 'neg': 0.0},
            {'pos': 0.752, 'compound': 0.8439, 'neu': 0.248, 'neg': 0.0},
            {'pos': 0.701, 'compound': 0.8545, 'neu': 0.299, 'neg': 0.0}
        ]

        EXPECTED = np.array(pd.DataFrame({"mockcolumn_sentiment": expected_values}))
        # EXPECTED = np.array(expected_values)

        self.log.warning("actual_feature", actual_feature)
        self.log.warning("EXPECTED", EXPECTED)

        self.assertEqual(actual_feature[0], EXPECTED[0])

        isEqual = np.array_equal(actual_feature, EXPECTED, equal_nan=False)
        self.assertTrue(isEqual)


if __name__ == '__main__':
    unittest.main()
