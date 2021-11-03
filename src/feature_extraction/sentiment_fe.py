#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
from nltk import tokenize
from src.feature_extraction.feature_extractor import FeatureExtractor


class SentimentFE(FeatureExtractor):
    """
    Analyzes the sentiment of a given text input column (e.g. the sentiment of a tweet).
    The result is outputted as two columns (positive and negative sentiment).
    """
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], f"{input_column}_sentiment")


    # don't need to fit, so don't overwrite _set_variables()


    def _get_values(self, inputs: pd.Series):

        analyzer = SentimentIntensityAnalyzer()
        # result = inputs[0].apply(lambda x: [analyzer.polarity_scores(sentence) for sentence in x])
        sentiments = inputs[0].apply(lambda x: analyzer.polarity_scores(x))

        temp_df = pd.DataFrame.from_records(sentiments)
            
        collected_rows = []
        for _, row in temp_df.iterrows():
            temp_row = [
                row["pos"], # positive
                row["neg"], # negative
                # other values are left out, 
                # because they contain no new information
                # row["compound"], # sum of pos and neu
                # row["neu"], # neutral
            ]
            collected_rows.append(temp_row)
                
        result = np.array(collected_rows)
               
        return result
