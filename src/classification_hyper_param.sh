#!/bin/bash

# perform grid search over the range of parameters

# k nearest neighbour classifier
values_of_k=("1 2") # ("1 2 3 4 5 6 7 8 9 10")
for k in $values_of_k
do
    echo $k
    RUN_NAME="knn with k=${k}"

    python -m src.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle -s 42 --knn $k --metrics all -n "${RUN_NAME}"    
    python -m src.classification.run_classifier data/dimensionality_reduction/validation.pickle -i data/classification/classifier.pickle --metrics all -n "${RUN_NAME}"
done

# random forest classifier
values_of_n=("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20")
for n in $values_of_n
do
    echo $n
    RUN_NAME="RF with n=${n}"

    python -m src.classification.run_classifier data/dimensionality_reduction/training.pickle -e data/classification/classifier.pickle -s 42 --randomforest $n --metrics all -n "${RUN_NAME}"
    python -m src.classification.run_classifier data/dimensionality_reduction/validation.pickle -i data/classification/classifier.pickle --metrics all -n "${RUN_NAME}"
done
