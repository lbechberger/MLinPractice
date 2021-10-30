# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:08:05 2021

@author: Yannik
"""

import spacy
import en_core_web_sm
import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

class NER(FeatureExtractor):
    """"""""
    
    def __init__(self, input_column):
       super().__init__([input_column], "{0}_ner_count".format(input_column))
       
    def _get_values(self, inputs):
        result = []
        nlp = en_core_web_sm.load()
        
        for tweet in inputs[0]:
            doc = nlp(tweet)
            result.append(len(doc.ents))
            
        result = np.array(result)
        return result.reshape(-1,1)
                
        