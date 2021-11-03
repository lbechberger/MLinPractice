#!/bin/bash

# create directory if not yet existing
mkdir -p data/classification/

# run feature extraction on training set (may need to fit extractors)

# echo "  training set"

RUN_NAME="after sentiment was added"
# uncomment one of these lines to train a specific classifier
python -m src.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle -s 42 --knn 1 --metrics all -n "${RUN_NAME}"
# python -m src.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle -s 42 --randomforest 10 --metrics all -n "${RUN_NAME}"
# python -m src.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle -s 42 --dummyclassifier stratified --metrics all -n "${RUN_NAME}"
# python -m src.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle -s 42 --dummyclassifier most_frequent --metrics all -n "${RUN_NAME}"

# run feature extraction on validation set (with pre-fit extractors)
# echo "  validation set"
python -m src.classification.run_classifier data/dimensionality_reduction/validation.pickle -i data/classification/classifier.pickle --metrics all

# don't touch the test set, yet, because that would ruin the final generalization experiment!