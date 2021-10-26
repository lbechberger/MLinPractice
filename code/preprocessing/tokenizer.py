#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenize the tweet into individual words.

Created on Wed Oct  6 13:59:54 2021

@author: lbechberger/rfarah
"""

from code.preprocessing.preprocessor import Preprocessor
import string
import nltk


class Tokenizer(Preprocessor):
    """Tokenizes the given input column into individual words."""
<<<<<<< HEAD

    def __init__(self, input_column, output_column):
=======
    
    def __init__(self, input_column, output_column): 
>>>>>>> 49c39fa (resolve the conflict)
        """Initialize the Tokenizer with the given input and output column."""
        super().__init__([input_column], output_column)

    def _set_variables(self, inputs):
        self.urls = ["http", "https", "www"]
        self.special_characters = ['@', '#']
        self.punctuation = [x for x in string.punctuation[1:-1]]

    def _get_values(self, inputs):
        """Tokenize the tweet."""

<<<<<<< HEAD
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
>>>>>>> 49c39fa (resolve the conflict)
        tokenized = []
        for tweet in inputs[0]:
            sentences = nltk.sent_tokenize(tweet.lower())
            tokenized_tweet = []
            for sentence in sentences:
                words = nltk.word_tokenize(sentence)
<<<<<<< HEAD
                words = self._delete_mentions(words)
                remove_urls_hashtags_mentions = [
                    word for word in words if word[0] not in self.punctuation and word[:4] not in self.urls and word[:3] not in self.urls]

                tokenized_tweet += remove_urls_hashtags_mentions

            tokenized.append(tokenized_tweet)

        return tokenized

    def _delete_mentions(self, words):
        """Deletes the hashtags, the mentions and numbers"""

        no_mentions = []

        for i in range(len(words)):
            if words[i][0] not in self.special_characters:
                if i > 0:
                    if words[i-1][0] not in self.special_characters and words[i].lower().islower():
                        no_mentions.append(words[i])
                elif i == 0 and words[i].lower().islower():
                    no_mentions.append(words[i])

        return no_mentions
=======
                tokenized_tweet += words
            
            tokenized.append(tokenized_tweet)

        return tokenized
>>>>>>> 49c39fa (resolve the conflict)
