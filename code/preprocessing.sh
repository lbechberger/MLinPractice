#!/bin/bash

# create directory if not yet existing
mkdir -p data/preprocessing/

# add labels
python -m code.preprocessing.create_labels data/raw/ data/preprocessing/labeled.csv

# other preprocessing (removing punctuation etc.)
python -m code.preprocessing.run_preprocessing

# split the data set
python -m code.preprocessing.split_data