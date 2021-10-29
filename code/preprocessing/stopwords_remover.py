#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that removes stopwords and punctuations from the original tweet text depending on the language.

Created on Wed Sep 29 09:45:56 2021

@author: rfarah
"""
import contractions
from nltk.corpus import stopwords
from code.preprocessing.preprocessor import Preprocessor

class StopwordsRemover(Preprocessor):
    # constructor
    def __init__(self, input_column, output_column): 
        # input column "tweet", new output column
        super().__init__([input_column], output_column)

    # set internal variables based on input columns
    def _set_variables(self, inputs):
        # load the punctuation list and the stopword lists for the following four languages
        self._stopwords_en = stopwords.words('english')
        self._stopwords_de = stopwords.words('german')
        self._stopwords_fr = stopwords.words('french')
        self._stopwords_es = stopwords.words('spanish')
    
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        inputs = inputs[0]
        # replace stopwords and punctuations with empty string
        tweets_withno_stopwords = []
        language = inputs.iloc[:,1][0]
        for tweet in inputs.iloc[:,0]:
            tweet = tweet[0]
            if tweet:
                tweet_withno_stopwords = []
                if 'en' in language:
                    tweet_withno_contractions = [contractions.fix(word).lower() for word in tweet]
                    tweet_withno_stopwords = [word.split(" ") for word in tweet_withno_contractions if word not in self._stopwords_en]
                    tweet_withno_stopwords_splitted = [splitted if isinstance(word, list) else word for word in tweet_withno_stopwords for splitted in word]
                    tweet_withno_stopwords = [word for word in tweet_withno_stopwords_splitted if word not in self._stopwords_en]
                elif 'de' in language:
                    tweet_withno_stopwords = [word for word in tweet if word.lower() not in self._stopwords_de]
                elif 'fr' in language:
                    tweet_withno_stopwords = [word for word in tweet if word.lower() not in self._stopwords_fr]
                elif 'es' in language:
                    tweet_withno_stopwords = [word for word in tweet if word.lower() not in self._stopwords_es]
                tweets_withno_stopwords.append(tweet_withno_stopwords)
        return tweets_withno_stopwords