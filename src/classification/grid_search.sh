#!/bin/bash

mkdir -p data/classification

# specify hyperparameter values
values_of_k=("1 2 3 4 5 6 7 8 9 10")
values_of_C=("0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0")


# different execution modes
if [ $1 = local ]
then
    echo "[local execution]"
    cmd="src/classification/classifier.sge"
elif [ $1 = grid ]
then
    echo "[grid execution]"
    cmd="qsub src/classification/classifier.sge"
else
    echo "[ERROR! Argument not supported!]"
    exit 1
fi

# do the grid search
# KNN
for k in $values_of_k
do
    echo $k
    $cmd 'data/classification/clf_knn_'"$k"'.pickle' --knn $k -s 42 --accuracy --kappa --precision --recall --f1_score
done

# LSVM
for C in $values_of_C
do
    echo "LSVM - $C"
    $cmd 'data/classification/clf_lsvm_'"$C"'.pickle' --lsvm $C -s 42 --accuracy --kappa --precision --recall --f1_score
done

# Gaussian Naive Bayes
echo "GNB"
$cmd 'data/classification/clf_gnb.pickle' --gnb -s 42 --accuracy --kappa --precision --recall --f1_score

# MLP
echo "MLP"
$cmd 'data/classification/clf_mlp.pickle' --mlp -s 42 --accuracy --kappa --precision --recall --f1_score

