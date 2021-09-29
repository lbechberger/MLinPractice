#!/bin/bash

echo "loading data"
code/load_data.sh
echo "preprocessing"
code/preprocessing.sh
echo "feature extraction"
code/feature_extraction.sh
echo "dimensionality reduction"
code/dimensionality_reduction.sh
echo "classification"
code/classification.sh