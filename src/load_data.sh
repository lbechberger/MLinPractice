#!/bin/bash

# create directory if not yet existing
mkdir -p data/raw/

# download the three csv files
wget -nv https://myshare.uni-osnabrueck.de/f/3e5276caf72b46e7ace2/?dl=1 -O data/raw/data_analysis.csv
wget -nv https://myshare.uni-osnabrueck.de/f/e620aff7719948d18a52/?dl=1 -O data/raw/data_science.csv
wget -nv https://myshare.uni-osnabrueck.de/f/9ddaab064c68483e9bff/?dl=1 -O data/raw/data_visualization.csv

