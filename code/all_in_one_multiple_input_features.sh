#!/bin/bash

# create directory if not yet existing
mkdir -p data/all_in_one_multiple_input_features/

# run feature extraction on training set (may need to fit extractors)
#echo "  training set"
#python3 -m code.all_in_one data/feature_extraction/training.pickle -e data/classification/classifier.pickle --accuracy --kappa --balanced_accuracy --classification_report --small 10000

# raw input, mit preprocessing
#python3 -m code.all_in_one data/preprocessing/split/training.csv -e data/classification/classifier.pickle --accuracy --kappa --balanced_accuracy --classification_report --hash_vectorizer #--count_vectorizer

# raw input, ohne preprocessing
#python3 -m code.all_in_one data/preprocessing/labeled.csv -e data/classification/classifier.pickle --accuracy --kappa --balanced_accuracy --classification_report --count_vectorizer #--hash_vectorizer #

# sklearn example
#python3 -m code.example_sklearn_pipeline data/preprocessing/split/training.csv


# run feature extraction on validation set (with pre-fit extractors)
#echo "  validation set"
#python3 -m code.all_in_one data/feature_extraction/validation.pickle -i data/classification/classifier.pickle --accuracy --kappa --balanced_accuracy --small 10000

# don't touch the test set, yet, because that would ruin the final generalization experiment!

# new approach
python3 -m code.all_in_one_multiple_input_features data/preprocessing/preprocessed.csv -e data/classification/classifier.pickle --accuracy --kappa --balanced_accuracy --classification_report --classifier 'SGDClassifier' --feature_extraction 'union' #--small 2000 #--balance 'over_sampler' # | HashingVectorizer TfidfVectorizer | SVC SGDClassifier LogisticRegression LinearSVC MultinomialNB data/preprocessing/split/training.csv data/preprocessing/labeled.csv data/preprocessing/preprocessed.csv