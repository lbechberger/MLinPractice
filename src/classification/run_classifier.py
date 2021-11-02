#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.
"""

import argparse, pickle, os
from pathlib import Path
from typing import Any, Callable, List, Tuple
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from mlflow import log_metric, log_param, set_tracking_uri

from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score, jaccard_score

METR_ACC = "accuracy"
METR_KAPPA = "kappa"
METR_PREC = "precision"
METR_REC = "recall"
METR_F1 = "f1"
METR_JAC = "jaccard"

def main():
    # setting up CLI
    parser = argparse.ArgumentParser(description = "Classifier")
    parser.add_argument("input_file", help = "path to the input pickle file")
    parser.add_argument("-s", '--seed', type = int, help = "seed for the random number generator", default = None)
    parser.add_argument("-e", "--export_file", help = "export the trained classifier to the given location", default = None)
    parser.add_argument("-i", "--import_file", help = "import a trained classifier from the given location", default = None)

    parser.add_argument("-d", '--dummyclassifier', choices=["most_frequent", "stratified"], default=None)
    parser.add_argument("--knn", type = int, help = "k nearest neighbor classifier with the specified value of k", default = None)
    parser.add_argument("-r", "--randomforest", type = int, help = "Random Forest classifier with the specified number of estimators (trees)", default = None)
    
    metrics_choices = ["none", "all", METR_ACC, METR_KAPPA, METR_PREC, METR_REC, METR_F1, METR_JAC]
    parser.add_argument("-m", "--metrics", choices=metrics_choices,  default="none")

    parser.add_argument("--log_folder", help = "where to log the mlflow results", default = "data/classification/mlflow")
    args = parser.parse_args()

    # load data
    with open(args.input_file, 'rb') as f_in:
        data = pickle.load(f_in)

    if args.log_folder is not None:
        set_tracking_uri(args.log_folder)

    if args.import_file is not None:
        # import a pre-trained classifier
        with open(args.import_file, 'rb') as f_in:
            input_dict = pickle.load(f_in)
        
        classifier = input_dict["classifier"]
        for param, value in input_dict["params"].items():
            log_param(param, value)
        
        log_param("dataset", "validation")

    else:   
        # manually set up a classifier        
        if args.dummyclassifier == "most_frequent":
            # majority vote classifier
            print("    always most_frequent label (Dummy Classifier)")
            log_param("classifier", "most_frequent")
            params = {"classifier": "most_frequent"}
            classifier = DummyClassifier(strategy = "most_frequent", random_state = args.seed)

        elif args.dummyclassifier == "stratified":
            # label frequency classifier
            print("    label frequency classifier")
            log_param("classifier", "stratified")
            params = {"classifier": "stratified"}
            classifier = DummyClassifier(strategy = "stratified", random_state = args.seed)
        
        elif args.randomforest is not None:
            print("    random forest classifier")
            log_param("classifier", "randomforest")
            log_param("n", args.randomforest)
            params = {"classifier": "randomforest", "n": args.randomforest}
            classifier = RandomForestClassifier(n_estimators = args.randomforest, random_state = args.seed)

        elif args.knn is not None:
            print("    {0} nearest neighbor classifier".format(args.knn))
            log_param("classifier", "knn")
            log_param("k", args.knn)
            params = {"classifier": "knn", "k": args.knn}
            standardizer = StandardScaler()
            knn_classifier = KNeighborsClassifier(args.knn, n_jobs = -1)
            classifier = make_pipeline(standardizer, knn_classifier)

        classifier.fit(data["features"], data["labels"].ravel())
        log_param("dataset", "training")

    prediction = classifier.predict(data["features"])
    
    evaluation_metrics = select_metrics_based_on_args(args.metrics)
    computed_metrics = compute_metrics(
        evaluation_metrics, data["labels"], prediction)
    
    print_input_file_name(args.input_file) # eg training set
    print_formatted_metrics(computed_metrics) # eg Accuracy: 0.908
    log_metrics(computed_metrics)
    # export the trained classifier if the user wants us to do so
    if args.export_file is not None:
        output_dict = {"classifier": classifier, "params": params}
        with open(args.export_file, 'wb') as f_out:
            pickle.dump(output_dict, f_out)


def print_input_file_name(input_file):
    print("      " + Path(input_file).stem + " set");        


def select_metrics_based_on_args(metrics: str):
    evaluation_metrics: List[Tuple[str, Callable[[Any, Any], float] ]] = []

    if metrics == METR_ACC or metrics == "all":
        evaluation_metrics.append(("Accuracy", accuracy_score))

    if metrics == METR_KAPPA or metrics == "all":
        evaluation_metrics.append(("Cohen_kappa", cohen_kappa_score))

    if metrics == METR_PREC or metrics == "all":
        evaluation_metrics.append(("Precision", precision_score))

    if metrics == METR_REC or metrics == "all":
        evaluation_metrics.append(("Recall", recall_score))

    if metrics == METR_F1 or metrics == "all":
        evaluation_metrics.append(("F1-Score", f1_score))

    if metrics == METR_JAC or metrics == "all":
        evaluation_metrics.append(("Jaccard", jaccard_score))

    return evaluation_metrics


def compute_metrics(evaluation_metrics, data_column, prediction):
    computed_metrics: List[Tuple[str, float]] = []

    for metric_name, metric in evaluation_metrics:
        metric_score = metric(data_column, prediction)
        computed_metrics.append((metric_name, metric_score))

    return computed_metrics


def print_formatted_metrics(computed_metrics):
    for metric_name, metric_score in computed_metrics:
        number_of_decimals = 3
        rounded_score = round(metric_score, number_of_decimals)
        print(f"\t{metric_name}: {rounded_score}")


def log_metrics(computed_metrics):
    for metric_name, metric_score in computed_metrics:
        log_metric(metric_name, metric_score)


if __name__ == "__main__":
    main()
