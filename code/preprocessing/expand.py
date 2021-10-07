#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor extends concatenations to their base words.

Created on Wed Oct 6 

@author: lmcdonald
"""

import string
import re
from code.preprocessing.preprocessor import Preprocessor
from code.preprocessing.util.contractions import CONTRACTION_MAP

# substitutes a variety of common contractions with their long form
# credit: https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72
class Expander(Preprocessor):

    # constructor
    def __init__(self, input_col, output):
        # input column "tweet", new output column
        super().__init__([input_col], output)
    
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs, contraction_mapping=CONTRACTION_MAP):
       
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)

        def expand_match(contraction):

            match = contraction.group(0)

            if contraction_mapping.get(match):
                expanded_contraction = contraction_mapping.get(match)
            else:
                expanded_contraction = contraction_mapping.get(match.lower())

            return expanded_contraction
        
        # replace contraction with long form
        column = inputs[0].str.replace(contractions_pattern, expand_match)
        return column