#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import string
from code.preprocessing.preprocessor import Preprocessor
from code.util import COLUMN_TWEET, COLUMN_PUNCTUATION
import pdb

punct = set(string.punctuation).union(string.digits).union("—")

# removes punctuation from the original tweet
# inspired by https://stackoverflow.com/a/45600350
class PunctuationRemover(Preprocessor):

    # constructor
    def __init__(self, inputcol, outputcol):
        # input column, new output column
        super().__init__([inputcol], outputcol)

    # set internal variables based on input columns
    def _set_variables(self, inputs):
        # store punctuation for later reference
        self._punctuation = "[{}]".format(
            string.punctuation + string.digits + "’" + "—" + "”" + "➡️" + "“" + "↓"
        )

    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        # replace punctuation with empty string
        column = inputs[0].str.replace(self._punctuation, "")
        return column
