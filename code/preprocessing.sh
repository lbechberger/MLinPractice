#!/bin/bash

# create directory if not yet existing
mkdir -p data/preprocessing/split/

# install all NLTK models
python -m nltk.downloader all

# add labels
echo "  creating labels"
python -m code.preprocessing.create_labels data/raw/ data/preprocessing/labeled.csv

# other preprocessing (removing punctuation etc.)
echo "  general preprocessing"
python -m code.preprocessing.run_preprocessing data/preprocessing/labeled.csv data/preprocessing/preprocessed.csv -s --stopwords_input "tweet" -p --punctuation_input "tweet_stopwords_removed" -t --tokenize_input "tweet_stopwords_removed_punct_removed" -l --lemmatize_input "tweet_stopwords_removed_punct_removed_tokenized" -e data/preprocessing/pipeline.pickle

# split the data set
echo "  splitting the data set"
python -m code.preprocessing.split_data data/preprocessing/preprocessed.csv data/preprocessing/split/ -s 42