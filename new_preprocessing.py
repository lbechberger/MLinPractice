#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 18:09:54 2021

@author: maximilian
"""

import pandas as pd
import csv
import string
import nltk

df = pd.read_csv("data/preprocessing/preprocessed.csv", quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n", low_memory=False)

tokenized = []

for tweet in df['tweet'][0:6]:
    sentences = nltk.sent_tokenize(tweet)
    tokenized_tweet = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tokenized_tweet += words
    
    tokenized.append(tokenized_tweet)
                
# --- 

def punctuation(rows):
    
    punct = set(string.punctuation) - {'#'}
    
    for row in rows:
        for x in row:
            if x in punct:
                row.remove(x)
                    
    return rows

def emoji(rows):

    for row in rows:
        for x in row:
            if x.startswith('U+'):
                x.encode('utf-16', 'surrogatepass')
                x.decode('utf-16')
                x.encode("raw_unicode_escape")
                x.decode("latin_1")
            

for x in df['tweet_tokenized'][8].replace('\\',''):
    if x.startswith('U+'):
        x.encode('utf-16', 'surrogatepass')
        x.decode('utf-16')
        x.encode("raw_unicode_escape")
        x.decode("latin_1")




