#!/bin/bash

# create directory if not yet existing
mkdir -p data/classification/

# run feature extraction on training set (may need to fit extractors)
echo "  training set"
python3 -m code.classification.run_classifier data/feature_extraction/training.pickle -e data/classification/classifier.pickle --LinearSVC --accuracy --kappa --balanced_accuracy --small 10000
# run feature extraction on validation set (with pre-fit extractors)
echo "  validation set"
python3 -m code.classification.run_classifier data/feature_extraction/validation.pickle -i data/classification/classifier.pickle --accuracy --kappa --balanced_accuracy

# don't touch the test set, yet, because that would ruin the final generalization experiment!
