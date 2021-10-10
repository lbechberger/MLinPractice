#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import pandas as pd
from code.feature_extraction.threads import Threads


class ThreadsTest(unittest.TestCase):

    def setUp(self):
        self._threads_extractor = Threads('tweet')

    def test_fit_transform(self):
        df = pd.DataFrame()
        df['tweet'] = [
            'This is a normal tweet',
            'This is a tweet with a thread',
            'This is a tweet with a 🧵',
            '1/4 Read on my dear',
            '1/ Read on my dear',
        ]

        output = self._threads_extractor.fit_transform(df)

        self.assertEqual(output[0, 0], False)
        self.assertEqual(output[1, 0], True)
        self.assertEqual(output[2, 0], True)
        self.assertEqual(output[3, 0], True)
        self.assertEqual(output[3, 0], True)

