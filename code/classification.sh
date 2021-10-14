#!/bin/bash

# create directory if not yet existing
mkdir -p data/classification/

# run feature extraction on training set (may need to fit extractors)
echo "  training set"
python -m code.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle -m --knn 5 --tree 5 -s 42 -a -k -f1

# run feature extraction on validation set (with pre-fit extractors)
echo "  validation set"
python -m code.classification.run_classifier data/dimensionality_reduction/validation.pickle -i data/classification/classifier.pickle -m -a -k -f1

# don't touch the test set, yet, because that would ruin the final generalization experiment!