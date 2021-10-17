#!/bin/bash

# create directory if not yet existing
mkdir -p data/all_in_one/

# run feature extraction on training set (may need to fit extractors)
echo "  training set"
#python3 -m code.all_in_one data/feature_extraction/training.pickle -e data/classification/classifier.pickle --accuracy --kappa --balanced_accuracy --classification_report --small 10000

# raw input, ohne preprocessing
python3 -m code.all_in_one data/preprocessing/split/training.csv -e data/classification/classifier.pickle --accuracy --kappa --balanced_accuracy --classification_report --hash_vectorizer #--count_vectorizer

# sklearn example
#python3 -m code.example_sklearn_pipeline data/preprocessing/split/training.csv


# run feature extraction on validation set (with pre-fit extractors)
echo "  validation set"
#python3 -m code.all_in_one data/feature_extraction/validation.pickle -i data/classification/classifier.pickle --accuracy --kappa --balanced_accuracy --small 10000

# don't touch the test set, yet, because that would ruin the final generalization experiment!
