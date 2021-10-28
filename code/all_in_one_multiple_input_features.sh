#!/bin/bash

# create directory if not yet existing
mkdir -p data/all_in_one_multiple_input_features/

# iterate through multiple classifiers if needed
classifieres=("LogisticRegression") #SGDClassifier LinearSVC LogisticRegression
for k in $classifieres; do
    echo $k
    # all_in_one_multiple_input_features.py gets as an input a csv file with your dataframe
    # we use here data/preprocessing/preprocessed.csv so you have to first run code/preprocessing.sh to download, preprocess and label the data correctly
    python3 -m code.all_in_one_multiple_input_features data/preprocessing/preprocessed.csv -e data/classification/classifier.pickle --accuracy --kappa --balanced_accuracy --classification_report --classifier $k --feature_extraction 'union' --verbose --small 2000 --dim_red 'SelectKBest(mutual_info_regression)' #  #--balance 'over_sampler' # | HashingVectorizer TfidfVectorizer | SVC SGDClassifier LogisticRegression LinearSVC MultinomialNB data/preprocessing/split/training.csv data/preprocessing/labeled.csv data/preprocessing/preprocessed.csv
done
