#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gensim.downloader as api
import pandas as pd
import ast
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the character-based length as a feature
class Word2Vec(FeatureExtractor):
    """
        Create Word2Vec feature.
        Read sklearn or our documentation for more information.
    """

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "tweet_word2vec")

    # don't need to fit, so don't overwrite _set_variables()

    def _get_values(self, inputs):

        embeddings = api.load(
            "word2vec-google-news-300"
        )  # Try glove-twitter-200 for classifier
        keywords = [
            "coding",
            "free",
            "algorithms",
            "statistics",
        ]  # deeplearning not present

        tokens = inputs[0].apply(
            lambda x: list(ast.literal_eval(x))
        )  # Column from Series to list

        similarities = []

        for rows in tokens:
            sim = []
            for word in keywords:
                for item in rows:
                    try:
                        sim.append(embeddings.similarity(item, word))
                    except KeyError:
                        pass
            # similarities.append(max(sim)-np.std(sim))
            similarities.append(round(np.mean(sim), 4))  # try median

        result = np.asarray(similarities)
        result = result.reshape(-1, 1)
        return result
