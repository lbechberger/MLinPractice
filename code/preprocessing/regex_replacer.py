#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that replaces text matched by a regex with a predefined text.

@author: marcelklehr
"""

import string
from code.preprocessing.preprocessor import Preprocessor
import re


class RegexReplacer(Preprocessor):
    
    # constructor
    def __init__(self, input_col, output, regex, replacement):
        # input column "tweet", new output column
        super().__init__([input_col], output)
        self._regex = regex
        self._replacement = replacement
    
    # set internal variables based on input columns -- not implemented
    
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        # replace regex matches with replacement string
        column = inputs[0].str.replace(self._regex, self._replacement, regex=True)
        return column