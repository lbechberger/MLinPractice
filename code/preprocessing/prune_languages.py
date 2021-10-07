#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lmcdonald
"""

import pandas as pd
import csv

class LanguagePruner():

    def __init__(self, df):
    
        self._df = df

    def get_language_counts(self):
        
        output_df = (self._df.language).groupby(self._df.language).count()
        output_df.to_csv('./data/preprocessing/language_counts.csv', index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")
        

    def drop_rows_by_language(self, language = "en"):
        
        output_df = self._df.drop(self._df[self._df.language != language].index)
        output_df.to_csv('./data/preprocessing/pruned.csv', index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")

        return output_df
        


    