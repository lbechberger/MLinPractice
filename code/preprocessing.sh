#!/bin/bash

# create directory if not yet existing
mkdir -p data/preprocessing/split/

# need to install all NLTK models
python -m nltk.dowloader all

# python -m nltk.downloader all

# add labels
echo "  creating labels"
python -m code.preprocessing.create_labels data/raw/ data/preprocessing/labeled.csv

# other preprocessing (removing punctuation etc.)
echo "  general preprocessing"
<<<<<<< HEAD
python -m code.preprocessing.run_preprocessing data/preprocessing/preprocessed.csv data/preprocessing/preprocessed.csv --tokenize --punctuation_removing --stopwords_removing --lemmatize -e data/preprocessing/pipeline.pickle
=======
python -m code.preprocessing.run_preprocessing data/preprocessing/preprocessed.csv data/preprocessing/preprocessed.csv --tokenize --punctuation_remover --stopwords_remover -e data/preprocessing/pipeline.pickle
>>>>>>> 49c39fa (resolve the conflict)

# split the data set
echo "  splitting the data set"
python -m code.preprocessing.split_data data/preprocessing/preprocessed.csv data/preprocessing/split/ -s 42