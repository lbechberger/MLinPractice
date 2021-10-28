
"""
# remove @user, # and link from tweet column
"""

import pandas as pd
import re
from src.preprocessing.preprocessor import Preprocessor

class TweetClean(Preprocessor):
    
    def __init__(self, input_column, output_column):
        """Initialize the Tokenizer with the given input and output column."""
        super().__init__([input_column], output_column)
    
    def _get_values(self, inputs):
        column = inputs[0]

        # removes hashtags
        # remove text after # with space at the end
        column = column.apply(lambda x : re.sub("#[A-Za-z0-9_\$\?\'\;\:\@\%\&\.\,]+\s","",x))
        # remove text starting with # and at the end of sentence
        column = column.apply(lambda x : re.sub("#[A-Za-z0-9_\$\?\'\;\:\@\%\&\.\,]+","",x))
        
        # removes username handles (e.g. @someusername)
        # remove text after @ with space at the end
        column = column.apply(lambda x : re.sub("@[A-Za-z0-9_\$\?\'\;\:\@\%\&\.\,]+\s+","",x))        
        #remove text starting with @ and at the end of sentence
        column = column.apply(lambda x : re.sub("@[A-Za-z0-9_\$\?\'\;\:\@\%\&\.\,]+","",x))
        
        # removes URLs starting with http or https
        column = column.apply(lambda x : re.sub("http\S+",r'',x))
        column = column.apply(lambda x : re.sub("http\S+\s",r'',x))
        
        # remove all non alphabet and non number to remove emojis encluding punctuation
        # we will not be needing the punctuation remover after this
        column = column.apply(lambda x : re.sub("[^a-zA-Z0-9 ]+","",x))
        
        # remove double spaces 
        column = column.apply(lambda x : re.sub("\s+"," ",x))

        # removes the space at the beginning and end of the sentence
        column = column.apply(lambda x : x.strip())

        return column

