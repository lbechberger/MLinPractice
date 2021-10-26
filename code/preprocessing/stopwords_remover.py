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
<<<<<<< HEAD
=======
from code.util import DETECTION_MODEL
>>>>>>> 49c39fa (resolve the conflict)



class StopwordsRemover(Preprocessor):
    # constructor
<<<<<<< HEAD
<<<<<<< HEAD
    def __init__(self, input_column, output_column):
=======
    def __init__(self, input_column, output_column): 
>>>>>>> 49c39fa (resolve the conflict)
=======
    def __init__(self, input_column, output_column):
>>>>>>> 35c7f58 (resolve the conflict)
        # input column "tweet", new output column
        super().__init__([input_column], output_column)

    # set internal variables based on input columns
    def _set_variables(self, inputs):
        # load the punctuation list and the stopword lists for the following four languages
<<<<<<< HEAD
        self._stopwords_ar = stopwords.words('arabic')
=======
>>>>>>> 49c39fa (resolve the conflict)
        self._stopwords_en = stopwords.words('english')
        self._stopwords_de = stopwords.words('german')
        self._stopwords_fr = stopwords.words('french')
        self._stopwords_es = stopwords.words('spanish')
<<<<<<< HEAD
<<<<<<< HEAD
        self._stopwords_da = stopwords.words('danish')
        self._stopwords_nl = stopwords.words('dutch')
        self._stopwords_hu = stopwords.words('hungarian')
        self._stopwords_it = stopwords.words('italian')
        self._stopwords_no = stopwords.words('norwegian')
        self._stopwords_pt = stopwords.words('portuguese')
        self._stopwords_ro = stopwords.words('romanian')
        self._stopwords_ru = stopwords.words('russian')
        self._stopwords_sv = stopwords.words('swedish')

=======
>>>>>>> 35c7f58 (resolve the conflict)

    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        inputs = inputs[0]
        # replace stopwords and punctuations with empty string
        tweets_withno_stopwords = []
        language = inputs.iloc[:, 1][0]
        for tweet in inputs.iloc[:, 0]:
            if isinstance(tweet, list) and len(tweet) == 1:
                tweet = tweet[0]
            tweet_withno_stopwords = []

            if 'en' in language:
                tweet_withno_contractions = [
                    contractions.fix(word) for word in tweet]
                tweet_withno_stopwords = [word.split(
                    " ") for word in tweet_withno_contractions if word.lower() not in self._stopwords_en]
                tweet_withno_stopwords_splitted = [splitted if isinstance(
                    word, list) else word for word in tweet_withno_stopwords for splitted in word]
                tweet_withno_stopwords = [
                    word for word in tweet_withno_stopwords_splitted if word.lower() not in self._stopwords_en]
            elif 'de' in language:
                tweet_withno_stopwords = [
                    word for word in tweet if word.lower() not in self._stopwords_de]
            elif 'fr' in language:
                tweet_withno_stopwords = [
                    word for word in tweet if word.lower() not in self._stopwords_fr]
<<<<<<< HEAD
            elif 'ar' in language:
                tweet_withno_stopwords = [
                    word for word in tweet if word.lower() not in self._stopwords_ar]
            elif 'es' in language:
                tweet_withno_stopwords = [
                    word for word in tweet if word.lower() not in self._stopwords_es]
            elif 'da' in language:
                tweet_withno_stopwords = [
                    word for word in tweet if word.lower() not in self._stopwords_da]
            elif 'nl' in language:
                tweet_withno_stopwords = [
                    word for word in tweet if word.lower() not in self._stopwords_nl]
            elif 'hu' in language:
                tweet_withno_stopwords = [
                    word for word in tweet if word.lower() not in self._stopwords_hu]
            elif 'it' in language:
                tweet_withno_stopwords = [
                    word for word in tweet if word.lower() not in self._stopwords_it]
            elif 'no' in language:
                tweet_withno_stopwords = [
                    word for word in tweet if word.lower() not in self._stopwords_no]
            elif 'pt' in language:
                tweet_withno_stopwords = [
                    word for word in tweet if word.lower() not in self._stopwords_pt]
            elif 'ro' in language:
                tweet_withno_stopwords = [
                    word for word in tweet if word.lower() not in self._stopwords_ro]
            elif 'ru' in language:
                tweet_withno_stopwords = [
                    word for word in tweet if word.lower() not in self._stopwords_ru]
            elif 'sv' in language:
                tweet_withno_stopwords = [
                    word for word in tweet if word.lower() not in self._stopwords_sv]

            tweets_withno_stopwords.append(tweet_withno_stopwords)
        return tweets_withno_stopwords
=======
    
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):

        # replace stopwords and punctuations with empty string
        tweets_withno_stopwords = []

        for tweet in inputs[0]:
            if tweet:
                tweet_withno_stopwords = []
                lang = DETECTION_MODEL.predict(tweet)[0][0][0]
                
                if 'en' in lang:
                    tweet_withno_contractions = [contractions.fix(word).lower() for word in tweet]
                    tweet_withno_stopwords = [word.split(" ") for word in tweet_withno_contractions if word not in self._stopwords_en]
                    tweet_withno_stopwords_splitted = [splitted if isinstance(word, list) else word for word in tweet_withno_stopwords for splitted in word]
                    tweet_withno_stopwords = [word for word in tweet_withno_stopwords_splitted if word not in self._stopwords_en]
                elif 'de' in lang:
                    tweet_withno_stopwords = [word for word in tweet if word not in self._stopwords_de]
                elif 'fr' in lang:
                    tweet_withno_stopwords = [word for word in tweet if word not in self._stopwords_fr]
                elif 'es' in lang:
                    tweet_withno_stopwords = [word for word in tweet if word not in self._stopwords_es]
                tweets_withno_stopwords.append(tweet_withno_stopwords)
        return tweets_withno_stopwords
>>>>>>> 49c39fa (resolve the conflict)
=======
            elif 'es' in language:
                tweet_withno_stopwords = [
                    word for word in tweet if word.lower() not in self._stopwords_es]

            tweets_withno_stopwords.append(tweet_withno_stopwords)
        return tweets_withno_stopwords
>>>>>>> 35c7f58 (resolve the conflict)
