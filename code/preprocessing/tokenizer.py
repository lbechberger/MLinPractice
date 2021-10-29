#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tokenize the tweet into individual words.
<<<<<<< HEAD
Created on Wed Oct  6 07:49:27 2021
=======

Created on Wed Oct  6 13:59:54 2021

<<<<<<< HEAD
>>>>>>> c9d63eb (created Tokenizer class)
@author: lbechberger
=======
@author: lbechberger/rfarah
>>>>>>> 0898e45 (add the lemmatizer and its test)
"""

from code.preprocessing.preprocessor import Preprocessor
import string
import nltk


class Tokenizer(Preprocessor):
    """Tokenizes the given input column into individual words."""
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

    def __init__(self, input_column, output_column):
        """Initialize the Tokenizer with the given input and output colum."""
=======
    
    def __init__(self, input_column, output_column): 
=======

    def __init__(self, input_column, output_column):
>>>>>>> a3333a8 (modified the tokenizer)
=======

    def __init__(self, input_column, output_column):
=======
    
    def __init__(self, input_column, output_column): 
>>>>>>> 49c39fa (resolve the conflict)
>>>>>>> 7105dbc (resolve the conflict)
=======

    def __init__(self, input_column, output_column):
>>>>>>> 828d2d3 (fix merge issues)
        """Initialize the Tokenizer with the given input and output column."""
>>>>>>> 2854caf (modified files and testing added)
        super().__init__([input_column], output_column)

<<<<<<< HEAD
    # don't need to implement _set_variables()

=======
    def _set_variables(self, inputs):
        self.urls = ["http", "https", "www"]
        self.special_characters = ['@', '#']
        self.punctuation = [x for x in string.punctuation[1:-1]]
<<<<<<< HEAD
>>>>>>> a3333a8 (modified the tokenizer)
=======

>>>>>>> 0898e45 (add the lemmatizer and its test)
    def _get_values(self, inputs):
        """Tokenize the tweet."""
<<<<<<< HEAD
        tokenized = []

=======

        tokenized = []
>>>>>>> 2854caf (modified files and testing added)
        for tweet in inputs[0]:
            sentences = nltk.sent_tokenize(tweet.lower())
            tokenized_tweet = []
            for sentence in sentences:
                words = nltk.word_tokenize(sentence)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
                tokenized_tweet += words
<<<<<<< HEAD

            tokenized.append(str(tokenized_tweet))

        return tokenized
=======
            
            tokenized.append(tokenized_tweet)

        return tokenized
>>>>>>> 2854caf (modified files and testing added)
=======
=======
>>>>>>> 7105dbc (resolve the conflict)
=======
>>>>>>> 828d2d3 (fix merge issues)
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> a3333a8 (modified the tokenizer)
=======
>>>>>>> 0898e45 (add the lemmatizer and its test)
=======
=======
                tokenized_tweet += words
            
            tokenized.append(tokenized_tweet)

        return tokenized
>>>>>>> 49c39fa (resolve the conflict)
>>>>>>> 7105dbc (resolve the conflict)
=======
>>>>>>> 828d2d3 (fix merge issues)
