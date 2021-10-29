#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides the lemmas of the words of the tweet using their POS.

inspired by https://www.machinelearningplus.com/nlp/lemmatization-examples-python/#wordnetlemmatizerwithappropriatepostag

@author: rfarah

"""

from code.preprocessing.preprocessor import Preprocessor
import nltk
import ast
from nltk.stem import WordNetLemmatizer


class Lemmatizer(Preprocessor):
    """Provides the lemmas of the words of the tweet using their POS. Works only for English"""

    def __init__(self, input_column, output_column):
        """Initialize the LemmatizeUsingPOS with the given input and output column."""
        super().__init__([input_column], output_column)

    def _get_wordnet_pos(self, tweet):
        pos = dict(nltk.pos_tag(tweet))
        # change the POSes so the lemmatizer can make use of them
        changed_pos = dict([(name, self.tag_dict.get(pos))
                           for name, pos in pos.items()])

        return changed_pos

    def _set_variables(self, inputs):
        self.lemmatizer = WordNetLemmatizer()
        self.tag_dict = {"MD": "a", "JJ": "a", "JJR": "a", "JJS": "a",
                         "NN": "n", "NNS": "n", "NNP": "n", "NNPS": "n", "FW": "n", "MD": "n",
                         "VB": "v", "VBD": "v", "VBG": "v", "VBN": "v", "VBP": "v", "VBZ": "v",
                         "RB": "r", "RBR": "r", "RBS": "r", "RP": "r"}

    def _get_values(self, inputs):
        """Finds the Lemmas of the tweet's words."""
        lemmatized = []

        for tweet in inputs[0]:
            if isinstance(tweet, str):
                tweet = ast.literal_eval(tweet)
            pos_tweet = self._get_wordnet_pos(tweet)
            row = [self.lemmatizer.lemmatize(word, pos) if pos else self.lemmatizer.lemmatize(
                word) for word, pos in pos_tweet.items()]
            lemmatized.append(row)

        return lemmatized
