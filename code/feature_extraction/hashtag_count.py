#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import ast
from code.feature_extraction.feature_extractor import FeatureExtractor


class HashtagCount(FeatureExtractor):
    """Count the number of hashtags in a sample."""

    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_count".format(input_column))

    # don't need to fit, so don't overwrite _set_variables()

    def _get_values(self, inputs):

        hashtags_list = inputs[0].astype(str).values.tolist()

        values = []
        for row in hashtags_list:
            if ast.literal_eval(row) == []:
                values.append(0)
            else:
                values.append(len(ast.literal_eval(row)))

        result = np.array(values)
        result = result.reshape(-1, 1)

        return result
