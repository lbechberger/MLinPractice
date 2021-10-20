#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of characters in the given column.

Created on Wed Sep 29 12:29:25 2021

@author: lbechberger
"""

import numpy as np
import gensim.downloader as api
import pandas as pd
import ast
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the character-based length as a feature
class Word2Vec(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_word2vec".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the word length based on the inputs
    def _get_values(self, inputs):
        
        embeddings = api.load('word2vec-google-news-300') # Try glove-twitter-200 for classifier 
        keywords = ['coding','free','algorithms','statistics'] # DataScience, BigData, deeplearning, machinelearning not present

        tokens = inputs[0].apply(lambda x: list(ast.literal_eval(x))) # Column from Series to list

        similarities = []
        
        for rows in tokens:
            sim = []
            for word in keywords:
                for item in rows:
                    try:
                        sim.append(embeddings.similarity(item,word))
                    except KeyError:
                        pass
            # similarities.append(max(sim)-np.std(sim))
            similarities.append(np.mean(sim)) # try median
    
        result = np.asarray(similarities)
        result = result.reshape(-1,1)
        return result
