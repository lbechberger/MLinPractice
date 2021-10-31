#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:51:00 2021

@author: ml
"""

import unittest
import pandas as pd
import numpy as np
from code.feature_extraction.popular_hashtags import PopularHashtags


class PopularHashtagsTest(unittest.TestCase):

    def setUp(self):
        self.INPUT_COLUMN = "hashtags"
        self.popular_hastags = PopularHashtags(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ["['datascientist', 'sasgf', 'machinelearning', 'bigdata', 'datascience','ai']",
                                      "['machinelearning' ,'datascience' ,'bigdata']",
                                      "['opendata', 'bigdata', 'datascience', 'rstats']",
                                      "['puertorico' ,'covid19', 'ai', 'datascience', 'python']",
                                      "['opendata', 'data', 'healthissues']",
                                      "['tech', 'cooltechjobs']",
                                      "['ai', 'datascience', 'ai4socialgood', 'data4good']",
                                      "['opensource', 'datavisualization']",
                                      "['datascientist', 'datascience', 'bigdata', 'machinelearning']"]

    def test_input_columns(self):
        self.assertEqual(self.popular_hastags._input_columns,
                         [self.INPUT_COLUMN])

    def test_most_related_hashtags(self):
        result = self.popular_hastags.fit_transform(self.df)
        most_popular = self.popular_hastags.most_popular_tweet[0]
        EXPECTED_MOST_POPULAR_HASHTAGS = 0
        self.assertEqual(most_popular, EXPECTED_MOST_POPULAR_HASHTAGS)


if __name__ == '__main__':
    unittest.main()
