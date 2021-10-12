#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
from code.feature_extraction.tf_idf import TfIdf


class TfIdfTest(unittest.TestCase):

    def setUp(self):
        self._tfidf_extractor = TfIdf('tweet')

    def test_fit_transform(self):
        df = pd.DataFrame()
        df['tweet'] = [
            'novel and not much else',
            'easy and not much else',
            'completely different',
        ]

        output = self._tfidf_extractor.fit_transform(df)
        # [and completely different easy else much not novel]

        self.assertLess(output[0, 0], 0.5)
        self.assertLess(output[0, 1], 0.5)
        self.assertLess(output[0, 2], 0.5)
        self.assertLess(output[0, 3], 0.5)
        self.assertLess(output[0, 4], 0.5)
        self.assertLess(output[0, 5], 0.5)
        self.assertLess(output[0, 6], 0.5)
        self.assertGreater(output[0, 7], 0.5)  # novel should be novel

        self.assertLess(output[1, 0], 0.5)
        self.assertLess(output[1, 1], 0.5)
        self.assertLess(output[1, 2], 0.5)
        self.assertGreater(output[1, 3], 0.5)  # easy should be novel
        self.assertLess(output[1, 4], 0.5)
        self.assertLess(output[1, 5], 0.5)
        self.assertLess(output[1, 6], 0.5)
        self.assertLess(output[1, 7], 0.5)

        self.assertLess(output[2, 0], 0.5)
        self.assertGreater(output[2, 1], 0.5)  # completely should be novel
        self.assertGreater(output[2, 2], 0.5)  # different should be novel
        self.assertLess(output[2, 3], 0.5)
        self.assertLess(output[2, 4], 0.5)
        self.assertLess(output[2, 5], 0.5)
        self.assertLess(output[2, 6], 0.5)
        self.assertLess(output[2, 7], 0.5)
