#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Checks the similarity between the sentence embeddings of the stemmed tweets
Created on Thu Oct  7 14:53:52 2021

@author: ml
"""

import ast
import sister
import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity


class SimilarTweets(FeatureExtractor):

    def __init__(self, input_column):
        super().__init__([input_column], "similar_{0}".format(input_column))

    def _get_values(self, inputs):
        inputs_concat = [[" ".join(o)] for x in inputs[0] if len(
            o := ast.literal_eval(x)) > 0]
        indices = [index for index, element in enumerate(inputs[0]) if element == '[]']
        length_of_nones = len(inputs[0]) - len(inputs_concat)
        embedder = sister.MeanEmbedding(lang="en")
        sentence_embeddings = [embedder(x[0]) for x in inputs_concat]

        start, bin_size = 0, 10000
        # to stop the Bus error: 10 and make the code computational wise more efficient
        if len(inputs_concat) > bin_size:
            end = bin_size
            bins = round(len(inputs_concat)/bin_size)
            final = []
            for i in range(bins):
                tmp_scores = cosine_similarity(
                    np.array(sentence_embeddings[start:end]))
                final.extend(np.sum(tmp_scores, axis= 1))
                print(final)
                start = end
                end += bin_size
        else:
            final = np.sum(cosine_similarity(np.array(sentence_embeddings)), axis=1)
        # add the NaN values to keep the correct shape
        for index in indices:
            final.insert(index, 0)
        final = final.reshape(-1, 1)
        return final
