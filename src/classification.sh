#!/bin/bash

# create directory if not yet existing
mkdir -p data/classification/




#-------------------------------------------------------------------#
# UNCOMMENT ONE OF THESE LINES BELOW TO TRAIN A SPECIFIC CLASSIFIER #
#-------------------------------------------------------------------#

RUN_NAME="2021-11-12 best params"

# python -m src.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle -s 42 --knn 1 --metrics all -n "${RUN_NAME}"
python -m src.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle -s 42 --randomforest 10 --metrics all -n "${RUN_NAME}"
# python -m src.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle -s 42 --randomforest 10 --metrics all -n "${RUN_NAME}" --sk_gridsearch_rf
# python -m src.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle -s 42 --dummyclassifier stratified --metrics all -n "${RUN_NAME}"
# python -m src.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle -s 42 --dummyclassifier most_frequent --metrics all -n "${RUN_NAME}"

# run feature extraction on validation set (with pre-fit classifier)
python -m src.classification.run_classifier data/dimensionality_reduction/validation.pickle -i data/classification/classifier.pickle --metrics all -n "${RUN_NAME}"

# don't touch the test set, yet, because that would ruin the final generalization experiment!