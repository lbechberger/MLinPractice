#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: lbechberger
"""

import pdb
import argparse
import pandas as pd
import numpy as np
import pickle

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    balanced_accuracy_score,
    classification_report,
)
from code.util  import KNN_K

def load_args():
    # setting up CLI
    parser = argparse.ArgumentParser(description="Classifier")
    parser.add_argument("input_file", help="path to the input pickle file")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="seed for the random number generator",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--export_file",
        help="export the trained classifier to the given location",
        default=None,
    )
    parser.add_argument(
        "-i",
        "--import_file",
        help="import a trained classifier from the given location",
        default=None,
    )
    parser.add_argument(
        "-m", "--majority", action="store_true", help="majority class classifier"
    )
    parser.add_argument(
        "-f", "--frequency", action="store_true", help="label frequency classifier"
    )
    parser.add_argument("-v", "--svm", action="store_true", help="SVM classifier")
    parser.add_argument("--SGDClassifier", action="store_true", help="SGD classifier")
    parser.add_argument(
        "--LogisticRegression", action="store_true", help="LogisticRegression"
    )
    parser.add_argument("--LinearSVC", action="store_true", help="LinearSVC")
    parser.add_argument("--MultinomialNB", action="store_true", help="MultinomialNB")

    parser.add_argument(
        "--knn",
        action="store_true",
        help="k nearest neighbor classifier with the specified value of k (in util.py",

    )
    parser.add_argument(
        "-a", "--accuracy", action="store_true", help="evaluate using accuracy"
    )
    parser.add_argument(
        "-k", "--kappa", action="store_true", help="evaluate using Cohen's kappa"
    )
    parser.add_argument(
        "--balanced_accuracy",
        action="store_true",
        help="evaluate using balanced_accuracy",
    )
    parser.add_argument(
        "--classification_report",
        action="store_true",
        help="evaluate using classification_report",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print information during training",
    )

    parser.add_argument(
        "--small", type=int, help="not use all data but just subset", default=None
    )
    parser.add_argument(
        "--balanced_data_set",
        action="store_true",
        help="arg for classifier, use balanced data",
    )

    args = parser.parse_args()
    return args


def load_dataset(args):
    """load a pickle file and reduce samples"""
    # load data
    with open(args.input_file, "rb") as f_in:
        data = pickle.load(f_in)

    # use less data to safe time for testing
    if args.small is not None:
        # if limit is given
        max_length = len(data["features"])
        limit = min(args.small, max_length)
        # go through data and limit it
        for key, value in data.items():
            data[key] = value[:limit]

    return data


def create_classifier(args, data):
    """Load or create a classifier with given args and sklearn methods."""
    # use balanced data in classifier
    balanced = "balanced" if args.balanced_data_set else None
    verbose = True if args.verbose else False
    if args.import_file is not None:
        # import a pre-trained classifier
        with open(args.import_file, "rb") as f_in:
            classifier = pickle.load(f_in)

    else:  # manually set up a classifier

        if args.majority:
            # majority vote classifier
            classifier = DummyClassifier(
                strategy="most_frequent", random_state=args.seed
            )
        elif args.frequency:
            # label frequency classifier
            classifier = DummyClassifier(strategy="stratified", random_state=args.seed)
        elif args.svm:
            classifier = make_pipeline(
                StandardScaler(), SVC(probability=True, verbose=verbose)
            )
        elif args.knn:
            print("    {0} nearest neighbor classifier".format(KNN_K))
            standardizer = StandardScaler()
            knn_classifier = KNeighborsClassifier(KNN_K, n_jobs=-1)
            classifier = make_pipeline(standardizer, knn_classifier)
        elif args.SGDClassifier:
            # standardizer = StandardScaler()
            classifier = SGDClassifier(
                    class_weight=balanced, random_state=args.seed, n_jobs=-1, verbose=verbose
                )
        elif args.MultinomialNB:
            classifier = MultinomialNB()
        elif args.LogisticRegression:
            standardizer = StandardScaler()
            classifier = LogisticRegression(
                class_weight=balanced, n_jobs=-1, random_state=args.seed, verbose=verbose#, max_iter=1000
            )
        elif args.LinearSVC:
            classifier = LinearSVC(
                class_weight=balanced, random_state=args.seed, verbose=verbose
            )

        try:
            classifier.fit(data["features"], data["labels"].ravel())
        except:
            raise UnboundLocalError("Import an classifier or choose one.")

    return classifier


def evaluate_classifier(args, data, prediction):
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

    if args.classification_report:
        categories = ["Flop", "Viral"]
        print(
            classification_report(data["labels"], prediction, target_names=categories)
        )



def export_classifier(args, classifier):
    # export the trained classifier if the user wants us to do so
    if args.export_file is not None:
        #pdb.set_trace()
        with open(args.export_file, "wb") as f_out:
            pickle.dump(classifier, f_out)


if __name__ == "__main__":
    args = load_args()

    data = load_dataset(args)

    classifier = create_classifier(args, data)
    # now classify the given data
    prediction = classifier.predict(data["features"])
    evaluate_classifier(args, data, prediction)

    export_classifier(args, classifier)
    
