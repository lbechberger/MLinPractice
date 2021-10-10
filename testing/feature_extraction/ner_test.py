#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unittester for the named-entity-recognition feature extractor.
Created on Sat Oct 9 23:59:37 2021
@author: louiskhub
"""

import unittest
import pandas as pd
from code.feature_extraction.ner import NER

class NERtest(unittest.TestCase):

    def setUp(self):
        self.ner = NER('tweet')

    def test_fit_transform_positive(self):
        df = pd.DataFrame()
        df['tweet'] = ['Lena went to the store and bought clothes.']
        output = self._ner_extractor.fit_transform(df)
        # under construction

if __name__ == '__main__':
    unittest.main() 