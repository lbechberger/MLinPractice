#!/bin/bash

# create directory if not yet existing
mkdir -p data/classification/


RUN_NAME="FINAL RUN KNN k=1"

python -m src.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle -s 42 --knn 1 --metrics none -n "${RUN_NAME}"

# run feature extraction on test set (with pre-fit classifier)
python -m src.classification.run_classifier data/dimensionality_reduction/test.pickle -i data/classification/classifier.pickle --metrics all -n "${RUN_NAME}"
python -m src.classification.run_classifier data/dimensionality_reduction/test.pickle -e data/classification/classifier.pickle -s 42 --dummyclassifier stratified --metrics all -n "${RUN_NAME}"

RUN_NAME="FINAL RUN random forest n=9"
python -m src.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle -s 42 --randomforest 9 --metrics none -n "${RUN_NAME}"

# run feature extraction on test set (with pre-fit classifier)
python -m src.classification.run_classifier data/dimensionality_reduction/test.pickle -i data/classification/classifier.pickle --metrics all -n "${RUN_NAME}"
python -m src.classification.run_classifier data/dimensionality_reduction/test.pickle -e data/classification/classifier.pickle -s 42 --dummyclassifier stratified --metrics all -n "${RUN_NAME}"