#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collects the feature values from many different feature extractors.

Created on Wed Sep 29 12:36:01 2021

@author: lbechberger
"""

import pdb
import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# extend FeatureExtractor for the sake of simplicity
class FeatureCollector(FeatureExtractor):

    # constructor
    def __init__(self, features):

        # store features
        self._features = features

        # collect input columns
        input_columns = []
        for feature in self._features:
            input_columns += feature.get_input_columns()

        # remove duplicate columns
        input_colums = list(set(input_columns))

        # call constructor of super class
        super().__init__(input_columns, "FeatureCollector")

    # overwrite fit: instead of calling _set_variables(), we forward the call to the features
    def fit(self, df):

        for feature in self._features:
            feature.fit(df)

    # overwrite transform: instead of calling _get_values(), we forward the call to the features
    def transform(self, df):

        all_feature_values = []

        for feature in self._features:
            # loop over each dimension and append it:
            # multi dimension feature -> create multiple columns
            for col in feature.transform(df).T:
                all_feature_values.append(col)

        result = np.array(all_feature_values).T
        return result

    def get_feature_names(self, df):
        feature_names = []
        for feature in self._features:
            dim = feature.transform(df).shape
            # append the name of a feature
            # one dimension feature
            if dim[1] == 1:
                feature_names.append(feature.get_feature_name())
            # multi dimension feature -> create multiple columns
            else:
                feature_names += [
                    feature.get_feature_name() + "_{0}".format(ind + 1)
                    for ind in range(dim[1])
                ]

        return feature_names
