#!/bin/bash

# create directory if not yet existing
#mkdir -p data/preprocessing/split/

# install all NLTK models
#python -m nltk.downloader all

# add labels
echo -e "\n -> creating labels\n"
python -m code.preprocessing.create_labels data/raw/ data/preprocessing/labeled.csv

# other preprocessing (removing punctuation etc.)
echo -e "\n -> general preprocessing\n"
python -m code.preprocessing.run_preprocessing data/preprocessing/labeled.csv data/preprocessing/preprocessed.csv --punctuation --stopwords --tokenize --language en -e data/preprocessing/pipeline.pickle

# split the data set
echo -e "\n -> splitting the data set\n"
python -m code.preprocessing.split_data data/preprocessing/preprocessed.csv data/preprocessing/split/ -s 42