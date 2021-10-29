#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenize the tweet into individual words.
<<<<<<< HEAD
Created on Wed Oct  6 07:49:27 2021
=======

Created on Wed Oct  6 13:59:54 2021

>>>>>>> c9d63eb (created Tokenizer class)
@author: lbechberger
"""

from code.preprocessing.preprocessor import Preprocessor
import nltk


class Tokenizer(Preprocessor):
    """Tokenizes the given input column into individual words."""
<<<<<<< HEAD

    def __init__(self, input_column, output_column):
        """Initialize the Tokenizer with the given input and output colum."""
=======
    
    def __init__(self, input_column, output_column): 
        """Initialize the Tokenizer with the given input and output column."""
>>>>>>> 2854caf (modified files and testing added)
        super().__init__([input_column], output_column)

    # don't need to implement _set_variables()

    def _get_values(self, inputs):
        """Tokenize the tweet."""
<<<<<<< HEAD
        tokenized = []

=======

        # all_input_tokenized = []
        # print(len(inputs), inputs[0])
        # if len(inputs) > 1:
        #     inputs = inputs[0]
        #     print(inputs)
        #     for tweet in inputs:
        #         tokenized = []
        #         sentences = nltk.sent_tokenize(tweet)
        #         for sentence in sentences:
        #             words = nltk.word_tokenize(sentence)
        #             tokenized.append(words)
        #     tokenized = [token for sublist in tokenized for token in sublist]
        #     all_input_tokenized.append(tokenized)
        # else:
        #     words = nltk.word_tokenize(inputs[0][0])

        #     all_input_tokenized.append(words)
        tokenized = []
>>>>>>> 2854caf (modified files and testing added)
        for tweet in inputs[0]:
            sentences = nltk.sent_tokenize(tweet.lower())
            tokenized_tweet = []
            for sentence in sentences:
                words = nltk.word_tokenize(sentence)
                tokenized_tweet += words
<<<<<<< HEAD

            tokenized.append(str(tokenized_tweet))

        return tokenized
=======
            
            tokenized.append(tokenized_tweet)

        return tokenized
>>>>>>> 2854caf (modified files and testing added)
