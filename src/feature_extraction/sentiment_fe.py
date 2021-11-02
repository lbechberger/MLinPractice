#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Counts the
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from nltk import tokenize
from src.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the character-based length as a feature
class SentimentFE(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], f"{input_column}_sentiment")


    # don't need to fit, so don't overwrite _set_variables()


    def _get_values(self, inputs):
        """
        Parses the string in every cell of the column/series as an array
        and counts the length in the cell of the output column
        """

        analyzer = SentimentIntensityAnalyzer()
        # result = inputs[0].apply(lambda x: [analyzer.polarity_scores(sentence) for sentence in x])
        result = inputs[0].apply(lambda x: analyzer.polarity_scores(x))

        """
        for sentence in inputs[0]:
            vs = analyzer.polarity_scores(sentence)
            print("{:-<65} {}".format(sentence, str(vs)))
        """
        #  {'pos': 0.746, 'compound': 0.8316, 'neu': 0.254, 'neg': 0.0}
        

        # The official documentation of vader recommend to split a paragraph into multiple sentence tokens
        # because vader works the best for sentences
        # then the average of the results is computed

        """
        sentence_list = tokenize.sent_tokenize(inputs[0])
        paragraphSentiments = 0.0
        for sentence in sentence_list:
            vs = analyzer.polarity_scores(sentence)
            print("{:-<69} {}".format(sentence, str(vs["compound"])))
            paragraphSentiments += vs["compound"]
        print("AVERAGE SENTIMENT FOR PARAGRAPH: \t" + str(round(paragraphSentiments / len(sentence_list), 4)))
        """
        # result = np.array(paragraphSentiments)
       
        result = np.array(result)
        result = result.reshape(-1,1)
        return result
