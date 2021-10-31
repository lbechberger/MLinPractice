#!/bin/bash

# create directory if not yet existing
mkdir -p data/classification/

# run feature extraction on training set (may need to fit extractors)
echo "  training set"
python -m code.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle --forest 20  -s 42 --accuracy --kappa -p -r -f1 # --majority --frequency --knn 5 # --forest 20 # --supportvm 2000 # --logistic 1000

# run feature extraction on validation set (with pre-fit extractors)
echo "  validation set"
python -m code.classification.run_classifier data/dimensionality_reduction/validation.pickle -i data/classification/classifier.pickle --accuracy --kappa -p -r -f1

# don't touch the test set, yet, because that would ruin the final generalization experiment!
#echo "  test set"
#python -m code.classification.run_classifier data/dimensionality_reduction/test.pickle -i data/classification/classifier.pickle --accuracy --kappa -p -r -f1
