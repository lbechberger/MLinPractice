#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:53:52 2021
@author: ml
"""

import ast
import nltk
import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor


class BigramFeature(FeatureExtractor):

    def __init__(self, input_column):
        super().__init__([input_column], "{0}_bigrams".format(input_column))

    def _set_variables(self, inputs):
        """Computes the frequency distribution of all bigrams in the tweets"""
        all_the_tweets = []
        for tweet in inputs[0]:
            tokens = ast.literal_eval(tweet)
            all_the_tweets += tokens
        self._bigrams_of_all_tweets = list(nltk.bigrams(all_the_tweets))
        self._freq_dist_of_all_tweets = nltk.FreqDist(
            self._bigrams_of_all_tweets)
        self._dictionary_of_all_tweets = {
            item[0]: item[1] for item in self._freq_dist_of_all_tweets.items()}

        self._freq_list = []
        for tweet in inputs[0]:
            tweet_bigram_freq = []
            tweet = ast.literal_eval(tweet)
            bigrams = list(nltk.bigrams(tweet))
            for bigram in bigrams:
                tweet_bigram_freq.append(
                    (bigram, self._dictionary_of_all_tweets.get(bigram)))
            # remove duplicates
            tweet_bigram_freq = list(
                filter(None, list(set(tweet_bigram_freq))))
            tweet_bigram_freq.sort(key=lambda x: x[1], reverse=True)
            self._freq_list.append(tweet_bigram_freq)

    def _get_values(self, inputs):
        all_the_tweets = []
        ## need to recompute the frequency distribution, case we have new input as the validation set
        for tweet in inputs[0]:
            tokens = ast.literal_eval(tweet)
            all_the_tweets += tokens
        self._bigrams_of_all_tweets = list(nltk.bigrams(all_the_tweets))
        self._freq_dist_of_all_tweets = nltk.FreqDist(
            self._bigrams_of_all_tweets)
        self._dictionary_of_all_tweets = {
            item[0]: item[1] for item in self._freq_dist_of_all_tweets.items()}

        self._freq_list = []
        for tweet in inputs[0]:
            tweet_bigram_freq = []
            tweet = ast.literal_eval(tweet)
            bigrams = list(nltk.bigrams(tweet))
            for bigram in bigrams:
                tweet_bigram_freq.append(
                    (bigram, self._dictionary_of_all_tweets.get(bigram)))

            # remove duplicates
            tweet_bigram_freq = list(
                filter(None, list(set(tweet_bigram_freq))))
            # print(tweet_bigram_freq)
            tweet_bigram_freq.sort(key=lambda x: x[1], reverse=True)
            frequencies = [frequency[1] for frequency in tweet_bigram_freq]
            # sum the frequencies
            self._freq_list.append(sum(frequencies))
        frequent = np.array(self._freq_list)
        return frequent.reshape(-1, 1)
