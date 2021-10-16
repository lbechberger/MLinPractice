#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: lbechberger
"""

import argparse, pickle
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# setting up CLI
parser = argparse.ArgumentParser(description = "Classifier")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("-s", '--seed', type = int, help = "seed for the random number generator", default = None)
parser.add_argument("-e", "--export_file", help = "export the trained classifier to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import a trained classifier from the given location", default = None)
parser.add_argument("-m", "--majority", action = "store_true", help = "majority class classifier")
parser.add_argument("-f", "--frequency", action = "store_true", help = "label frequency classifier")
parser.add_argument("-v", "--svm", action = "store_true", help = "SVM classifier")
parser.add_argument("--sgd", action = "store_true", help = "SGD classifier")
parser.add_argument("--knn", type = int, help = "k nearest neighbor classifier with the specified value of k", default = None)
parser.add_argument("-a", "--accuracy", action = "store_true", help = "evaluate using accuracy")
parser.add_argument("-k", "--kappa", action = "store_true", help = "evaluate using Cohen's kappa")
parser.add_argument("--balanced_accuracy", action = "store_true", help = "evaluate using balanced_accuracy")
parser.add_argument("--small", type = int, help = "not use all data but just subset", default = None)

args = parser.parse_args()
#args, unk = parser.parse_known_args()
# load data
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)

if args.import_file is not None:
    # import a pre-trained classifier
    with open(args.import_file, 'rb') as f_in:
        classifier = pickle.load(f_in)

else:   # manually set up a classifier
    
    if args.majority:
        # majority vote classifier
        print("    majority vote classifier")
        classifier = DummyClassifier(strategy = "most_frequent", random_state = args.seed)
    elif args.frequency:
        # label frequency classifier
        print("    label frequency classifier")
        classifier = DummyClassifier(strategy = "stratified", random_state = args.seed)
    elif args.svm:
        print("    SVM classifier")
        classifier = make_pipeline(StandardScaler(), SVC(n_jobs=-1, probability=True, verbose=1))
    elif args.knn is not None:
        print("    {0} nearest neighbor classifier".format(args.knn))
        standardizer = StandardScaler()
        knn_classifier = KNeighborsClassifier(args.knn, n_jobs=-1)
        classifier = make_pipeline(standardizer, knn_classifier)
    elif args.sgd:
        print("    sgd classifier")
        standardizer = StandardScaler()
        classifier = make_pipeline(standardizer, SGDClassifier(n_jobs=-1, verbose=1))


if args.small is not None:
    # if limit is given
    max_length = len(data['features'])
    limit = min(args.small, max_length)
    # go through data and limit it
    for key, value in data.items():
        data[key] = value[:limit]


classifier.fit(data["features"], data["labels"].ravel())
# now classify the given data
prediction = classifier.predict(data["features"])



# collect all evaluation metrics
evaluation_metrics = []
if args.accuracy:
    evaluation_metrics.append(("accuracy", accuracy_score))
if args.kappa:
    evaluation_metrics.append(("Cohen's kappa", cohen_kappa_score))
if args.balanced_accuracy:
    evaluation_metrics.append(("balanced accuracy", balanced_accuracy_score))
# compute and print them
for metric_name, metric in evaluation_metrics:
    print("    {0}: {1}".format(metric_name, metric(data["labels"], prediction)))
    
# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(classifier, f_out)
