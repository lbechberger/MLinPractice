#!/bin/bash

# create directory if not yet existing
mkdir -p data/classification/

classifieres=("SGDClassifier LinearSVC LogisticRegression majority MultinomialNB")
for k in $classifieres
do
    echo $k
    # run feature extraction on training set (may need to fit extractors)
    echo "  training set"
    python -m code.classification.run_classifier data/feature_extraction/training.pickle -e data/classification/classifier.pickle --accuracy --kappa --balanced_accuracy --classification_report --balanced_data_set --$k # --verbose # --LinearSVC SGDClassifier
    # run feature extraction on validation set (with pre-fit extractors)
    echo "  validation set"
    python -m code.classification.run_classifier data/feature_extraction/validation.pickle -i data/classification/classifier.pickle --accuracy --kappa --balanced_accuracy --classification_report # --small 10000
    echo "  Done \n"
    # don't touch the test set, yet, because that would ruin the final generalization experiment!
done