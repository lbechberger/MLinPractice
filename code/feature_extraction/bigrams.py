#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:53:52 2021
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> a7c7fdb (unit test and TDD example)
=======

=======
>>>>>>> 33bd5e7 (Update with origin (feature-preprocessing tests))
>>>>>>> a9f1bb1 (resolve merge conflicts)
=======
>>>>>>> 0a0cfa3 (changes in the bigram)
@author: ml
"""

import ast
import nltk
from code.feature_extraction.feature_extractor import FeatureExtractor


class BigramFeature(FeatureExtractor):
    
    
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_bigrams".format(input_column))

    
    def _set_variables(self, inputs):
        all_the_tweets = []
        for tweet in inputs:
            tokens = ast.literal_eval(tweet.item())
    
            all_the_tweets += tokens
        

        self._bigrams_of_all_tweets = nltk.bigrams(all_the_tweets)
        self._freq_dist = nltk.FreqDist(self._bigrams_of_all_tweets)
        self._dictionary_of_all_tweets = {item[0]:item[1] for item in self._freq_dist.items()}

        freq_list = []

        # for tweet in inputs:
        #     tweet_bigram_freq = []
        #     tweet = ast.literal_eval(tweet.item())
        #     bigrams = list(nltk.bigrams(tweet))

        #     for bigram in bigrams:
        #         tweet_bigram_freq.append((bigram,self._dictionary_of_all_tweets.get(bigram)))

        #     tweet_bigram_freq.sort(key=lambda x: x[1], reverse=True)
        #     print(tweet_bigram_freq)
            
        #     freq_list.append(tweet_bigram_freq)
        # return freq_list
 

    def _get_values(self, inputs):
        freq_list = []

        for tweet in inputs:
            tweet_bigram_freq = []
            bigrams = nltk.bigrams(tweet)
            for bigram in bigrams:
                tweet_bigram_freq.append((bigram,self._dictionary_of_all_tweets(bigram)))
            tweet_bigram_freq.sort(key=lambda x: x[1], reverse=True)
            
            freq_list.append(tweet_bigram_freq)
        print("here")
        return freq_list
