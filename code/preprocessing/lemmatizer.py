#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that reduces words to their lemmas.
@author: louiskhub
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


class Lemmatizer(Preprocessor):
    """Reduces words to their lemmas."""

    def __init__(self, input_col, output_col):
        """Initialize the Lemmatizer with the given input and output column."""
        super().__init__([input_col], output_col)
    
    # no need to implement _set_variables
    
    def _get_values(self, inputs):
        """Lemmatize the words."""
        
        lemmatized = []
        lemmatizer = WordNetLemmatizer()
        for token, tag in pos_tag(inputs):
            
            pos = tag[0].lower()
            # Check the token's Part-of-Speech Tag for better lemmatization
            if pos not in ['a' , 'r' , 'n' , 'v']:
                pos = 'n' # Default is 'Noun'
    
            lemma = lemmatizer.lemmatize(token,pos)
            lemmatized.append(lemma)
        
        return lemmatized