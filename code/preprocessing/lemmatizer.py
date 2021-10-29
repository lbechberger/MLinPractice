#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lemmatizes the given input column, i.e. modifies inflected or variant forms of a word into its lemma.

Created on Fri Oct  8 11:18:30 2021

@author: dhesenkamp
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from ast import literal_eval


class Lemmatizer(Preprocessor):
    """Lemmatize given input column."""
    
    
    def __init__(self, input_column, output_column):
        """Init with given input and output column"""
        super().__init__([input_column], output_column)
    
    
    # implementation of _set_variables() not necessary
    
    
    # inspired by https://www.machinelearningplus.com/nlp/lemmatization-examples-python/    
    def _get_values(self, inputs):
        """Lemmatize given input based on WordNet. Also changes to lowercase."""
        lemmatizer = WordNetLemmatizer()
        
        # dict to map PoS to arg accepted by lemmatize()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV
                    }
        column = []
        
        for tweet in inputs[0]:
            tweet_eval = literal_eval(tweet)
            lemmatized = []
            
            for word in tweet_eval:
                # get first letter of PoS tag to retrieve entry from dict
                tag = pos_tag([word])[0][1][0].upper()
                lemmatized.append(lemmatizer.lemmatize(word.lower(), tag_dict.get(tag, wordnet.NOUN)))
            column.append(lemmatized)
            
        return column
        