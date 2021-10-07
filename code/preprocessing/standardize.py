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
from code.preprocessing.util.spellings import SPELLINGS_MAP


# substitutes UK spellings with US form
# credit: http://www.tysto.com/uk-us-spelling-list.html

class Standardizer(Preprocessor):

    # constructor
    def __init__(self, input_col, output):
        # input column "tweet", new output column
        super().__init__([input_col], output)
    
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs, spellings=SPELLINGS_MAP):
    
        uk_spelling_pattern = re.compile('({})'.format('|'.join(spellings.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)

        def standardize(uk_spelling):
            
            match = uk_spelling.group(0)

            if spellings.get(match):
                us_spelling = spellings.get(match)
            else:
                us_spelling = spellings.get(match.lower())

            return us_spelling
        
        # change UK spelling to US spelling
        column = inputs[0].str.replace(uk_spelling_pattern, standardize)
        return column