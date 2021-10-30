# -*- coding: utf-8 -*-
"""
Named entitiy recognition using spaCy's corpus. 
Works with unprocessed tweet column as default input.

Created on Thu Oct 28 10:08:05 2021

@author: Yannik
modified by dhesenkamp
"""

import spacy
import en_core_web_sm
import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor


class NER(FeatureExtractor):
    """Named entity recognition as a count."""
    
    
    def __init__(self, input_column):
        """Constructor, calls super Constructor."""
        super().__init__([input_column], "{0}_ner_count".format(input_column))
    

    # don't need to fit, so don't overwrite _set_variables()
    
       
    def _get_values(self, inputs):
        """Recognize named entities and counts their occurence on a per tweet basis."""
        result = []
        nlp = en_core_web_sm.load()
        
        for tweet in inputs[0]:
            doc = nlp(tweet)
            result.append(len(doc.ents))
            
        result = np.array(result)
        result = result.reshape(-1, 1)
        
        return result 