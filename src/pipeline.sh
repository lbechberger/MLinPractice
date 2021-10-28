#!/bin/bash
# overall pipeline for the ML experiments

echo "preprocessing"
src/preprocessing.sh
echo "feature extraction"
src/feature_extraction.sh
echo "dimensionality reduction"
src/dimensionality_reduction.sh
echo "classification"
src/classification.sh