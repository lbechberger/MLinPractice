#!/bin/bash

# runs all classifier with the configurations we want to explore as part of
# the hyperparamter optimization

# create directory if not yet existing
mkdir -p data/classification/


# knn
for i in "uniform" "distance"
do
    for j in 1 3 5 7 9
    do 
        echo "  training set"
        python -m code.classification.run_classifier data/feature_extraction/training.pickle -e data/classification/classifier.pickle --knn $i $j -s 42 -a -k -f1 -ba
        echo "  validation set"
        python -m code.classification.run_classifier data/feature_extraction/validation.pickle -i data/classification/classifier.pickle -a -k -f1 -ba
    done
done


# decision tree
for i in "gini" "entropy"
do
    for j in 16 18 20 22 24 26 28 30 32
    do
        echo "  training set"
        python -m code.classification.run_classifier data/feature_extraction/training.pickle -e data/classification/classifier.pickle --tree --tree_criterion $i --tree_depth $j -s 42 -a -k -f1 -ba
        echo "  validation set"
        python -m code.classification.run_classifier data/feature_extraction/validation.pickle -i data/classification/classifier.pickle -a -k -f1 -ba
    done
done


# random forest
for i in "gini" "entropy"
do
    for j in 10 25 50 100
    do
        for k in 16 18 20 22 24 26 28 30 32
        do
            echo "  training set"
            python -m code.classification.run_classifier data/feature_extraction/training.pickle -e data/classification/classifier.pickle --randforest $j --forest_criterion $i --forest_max_depth $k -s 42 -a -k -f1 -ba
            echo "  validation set"
            python -m code.classification.run_classifier data/feature_extraction/validation.pickle -i data/classification/classifier.pickle -a -k -f1 -ba
        done
    done
done


# svm
for i in "linear" "poly" "rbf" "sigmoid"
do
    echo "  training set"
    python -m code.classification.run_classifier data/feature_extraction/training.pickle -e data/classification/classifier.pickle --svm $i -s 42 -a -k -f1 -ba
    echo "  validation set"
    python -m code.classification.run_classifier data/feature_extraction/validation.pickle -i data/classification/classifier.pickle -a -k -f1 -ba
done


# mlp
for i in 10 25 50
do
    for j in 10 25 50
    do
        for k in 10 25 50
        do
            echo "  training set"
            python -m code.classification.run_classifier data/feature_extraction/training.pickle -e data/classification/classifier.pickle --mlp $i $j $k -s 42 -a -k -f1 -ba
            echo "  validation set"
            python -m code.classification.run_classifier data/feature_extraction/validation.pickle -i data/classification/classifier.pickle -a -k -f1 -ba
        done
    done
done

# bayes
echo "  training set"
python -m code.classification.run_classifier data/feature_extraction/training.pickle -e data/classification/classifier.pickle --bayes -s 42 -a -k -f1 -ba
echo "  validation set"
python -m code.classification.run_classifier data/feature_extraction/validation.pickle -i data/classification/classifier.pickle -a -k -f1 -ba
