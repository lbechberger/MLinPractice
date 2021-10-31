#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: lbechberger
"""

import argparse, pickle
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from mlflow import log_metric, log_param, set_tracking_uri

# setting up CLI
parser = argparse.ArgumentParser(description = "Classifier")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("-s", '--seed', type = int, help = "seed for the random number generator", default = None)
parser.add_argument("-e", "--export_file", help = "export the trained classifier to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import a trained classifier from the given location", default = None)
parser.add_argument("-m", "--majority", action = "store_true", help = "majority class classifier")
parser.add_argument("-f", "--frequency", action = "store_true", help = "label frequency classifier")
parser.add_argument("-n", "--knn", type = int, help = "k nearest neighbor classifier with the specified value of k", default = None)
parser.add_argument("-rf", "--forest", type = int, help = "random forest classifier with the specified number of trees", default = None)
parser.add_argument("-svm", "--supportvm", type=int, help = "linear support vector machine with specified number of iterations", default = None)
parser.add_argument("-l", "--logistic", type = int, help = "logistic regression classifier with specified number of epochs", default = None)
parser.add_argument("-a", "--accuracy", action = "store_true", help = "evaluate using accuracy")
parser.add_argument("-p", "--precision", action = "store_true", help = "evaluate using precision")
parser.add_argument("-r", "--recall", action = "store_true", help = "evaluate using recall")
parser.add_argument("-f1", "--fscore", action = "store_true", help = "evaluate using F-score")
parser.add_argument("-k", "--kappa", action = "store_true", help = "evaluate using Cohen's kappa")
parser.add_argument("--log_folder", help = "where to log the mlflow results", default = "data/classification/mlruns")
args = parser.parse_args()

# load data
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)

# setup logging 
set_tracking_uri(args.log_folder)

# import a pre-trained classifier
if args.import_file is not None:
    
    with open(args.import_file, 'rb') as f_in:
        input_dict = pickle.load(f_in)
    
    classifier = input_dict["classifier"]
    for param, value in input_dict["params"].items():
        log_param(param, value)
    
    log_param("dataset", "validation")


else:   # manually set up a classifier
    
    if args.majority:
        # majority vote classifier
        print("    majority vote classifier")
        log_param("classifier", "majority")
        params = {"classifier": "majority"}
        classifier = DummyClassifier(strategy = "most_frequent", random_state = args.seed)
        
    elif args.frequency:
        # label frequency classifier
        print("    label frequency classifier")
        log_param("classifier", "frequency")
        params = {"classifier": "frequency"}
        classifier = DummyClassifier(strategy = "stratified", random_state = args.seed)

    elif args.knn is not None:
        # k nearest neighbor classifier 
        print("    {0} nearest neighbor classifier".format(args.knn))
        log_param("classifier", "knn")
        log_param("k", args.knn)
        params = {"classifier": "knn", "k": args.knn}
        standardizer = StandardScaler()
        knn_classifier = KNeighborsClassifier(args.knn, n_jobs = -1)
        classifier = make_pipeline(standardizer, knn_classifier)

    elif args.forest is not None:
        # random forest classifier 
        print("     random forest classifier with {0} trees".format(args.forest)) # default 100
        log_param("classifier", "random_forest")
        log_param("n_trees", args.forest)
        params = {"classifier":"random_forest", "n_trees": args.forest}
        classifier = RandomForestClassifier(n_estimators = args.forest)

    elif args.supportvm is not None:
        # linear svm with specified number of iterations
        print("     linear support vector machine with {0} iterations".format(args.supportvm)) # default 1000 
        log_param("classifier", "support_vector_machine")
        log_param("iterations", args.supportvm)
        params = {"classifier":"support_vector_machine", "iterations": args.supportvm}
        classifier = LinearSVC(max_iter=args.supportvm)

    elif args.logistic is not None:
        # logistic regression classifier with stochastic gradient descent training
        print("     logistic regression classifier trained with stochastic gradient descent with {0} epochs".format(args.logistic)) # default 1000
        log_param("classifier", "logistic_regression")
        log_param("epochs", args.logistic)
        params = {"classifier":"logistic_regression", "epochs": args.logistic}
        classifier = SGDClassifier(loss='log', max_iter = args.logistic)
 

    # fit classifier to data 
    classifier.fit(data["features"], data["labels"].ravel())
    log_param("dataset", "training")

# now classify the given data
prediction = classifier.predict(data["features"])

# collect all evaluation metrics
evaluation_metrics = []
if args.accuracy:
    evaluation_metrics.append(("accuracy", accuracy_score))

if args.kappa:
    evaluation_metrics.append(("Cohen_kappa", cohen_kappa_score))

if args.fscore:
    evaluation_metrics.append(("F1 score",f1_score))

if args.precision:
    evaluation_metrics.append(("precision", precision_score))

if args.recall:
    evaluation_metrics.append(("recall", recall_score))

# compute and print them
for metric_name, metric in evaluation_metrics:
    metric_value = metric(data["labels"], prediction)
    print("    {0}: {1}".format(metric_name, metric_value))
    log_metric(metric_name, metric_value)
    
# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    output_dict = {"classifier": classifier, "params": params}
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(output_dict, f_out)
