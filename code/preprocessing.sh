#!/bin/bash

# create directory if not yet existing
mkdir -p data/preprocessing/split/

# install all NLTK models
# python -m nltk.downloader all

# # install the spaCy model for Englisch, French, Germana and Spanish
# python -m spacy download en_core_web_sm
# python -m spacy download fr_core_news_sm
# python -m spacy download de_core_news_sm
# python -m spacy download es_core_news_sm

# add labels
echo "  creating labels"
python -m code.preprocessing.create_labels data/raw/ data/preprocessing/labeled.csv

# other preprocessing (removing punctuation etc.)
echo "  general preprocessing"
python -m code.preprocessing.run_preprocessing data/preprocessing/labeled.csv data/preprocessing/preprocessed.csv --tokenize --punctuation_removing --stopwords_removing --stemming -e data/preprocessing/pipeline.pickle

# split the data set
echo "  splitting the data set"
python -m code.preprocessing.split_data data/preprocessing/preprocessed.csv data/preprocessing/split/ -s 42