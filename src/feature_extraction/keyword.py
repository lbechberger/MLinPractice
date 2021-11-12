#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyword class for feature extractor

Created: 11.11.21, 22:47

Author: LDankert
"""


import pandas as pd
from nltk import FreqDist
from sklearn.preprocessing import MultiLabelBinarizer
from src.feature_extraction.feature_extractor import FeatureExtractor
from src.util import COLUMN_LABEL, COLUMN_TWEET_CLEANED


# class for extracting the most common keywords in viral tweets only
class Keyword(FeatureExtractor):

    # constructor
    def __init__(self, input_columns, number_of_keywords):
        super().__init__(input_columns, "keywords")
        self.number_of_keywords = number_of_keywords

    # set the variables depending on the number of keywords
    def _set_variables(self, inputs):
        # creating a dataframe out of the cleaned_tweet and label column
        df = pd.DataFrame(list(zip(inputs[0], inputs[1])), columns=[COLUMN_TWEET_CLEANED, COLUMN_LABEL])

        # only take the tweets into account that were viral
        df_filtered = df.loc[df[COLUMN_LABEL] == True]
        cleaned_tweets = df_filtered[COLUMN_TWEET_CLEANED].tolist()

        all_tweets = []
        for tweet in cleaned_tweets:
            all_tweets.extend(tweet)

        freq = FreqDist(all_tweets)
        self.keywords = freq.most_common(self.number_of_keywords)
        print(f"    The {self.number_of_keywords} most viral keywords:")
        for keyword in self.keywords:
            print(f"        {keyword[0]}: {keyword[1]}")

    # returns columns, one for each most common words one
    def _get_values(self, inputs):
        result = []
        keywords = [keyword for keyword, _ in self.keywords]
        for tweet in inputs[0]:
            keywords_in_tweet = []
            keywords_in_tweet += [keyword for keyword in keywords if keyword in tweet]
            result.append(keywords_in_tweet)

        enc = MultiLabelBinarizer()
        result = enc.fit_transform(result)
        return result
