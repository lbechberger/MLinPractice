#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.
"""

import argparse, pickle
from pathlib import Path
from typing import Any, Callable, List, Tuple
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import mlflow
from mlflow import log_metric, log_param, set_tracking_uri, start_run
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score
from sklearn.metrics import recall_score, f1_score, jaccard_score, make_scorer
import pandas as pd
import numpy as np

METR_ACC = "accuracy"
METR_KAPPA = "kappa"
METR_PREC = "precision"
METR_REC = "recall"
METR_F1 = "f1"
METR_JAC = "jaccard"

def main():

    args = parse_arguments()

    # load data
    with open(args.input_file, 'rb') as f_in:
        data = pickle.load(f_in)

    set_tracking_uri(args.log_folder)
    with mlflow.start_run(run_name = args.run_name):

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
                
            elif args.knn is not None:
                print("    {0} nearest neighbor classifier".format(args.knn))
                log_param("classifier", "knn")
                log_param("k", args.knn)
                params = {"classifier": "knn", "k": args.knn}
                standardizer = StandardScaler()
                knn_classifier = KNeighborsClassifier(args.knn, n_jobs = -1)
                classifier = make_pipeline(standardizer, knn_classifier)

            elif args.randomforest is not None:

                if args.sk_gridsearch_rf is None:
                    print("    random forest classifier")
                    log_param("classifier", "randomforest")
                    log_param("n", args.randomforest)
                    params = {"classifier": "randomforest", "n": args.randomforest}
                    # classifier = RandomForestClassifier(n_estimators = args.randomforest, random_state = args.seed)                   
                                        
                    classifier = RandomForestClassifier(
                        criterion= 'entropy',
                        n_estimators = args.randomforest,
                        min_samples_split=2,
                        random_state = args.seed)

                else:
                    print("    grid search for random forest classifier")
                    estim_range = np.arange(5, 130, 12).tolist()
                    parameters = {
                        'n_estimators': estim_range,
                        'min_samples_split': [2,4,6,8]
                    }
                    scoring = {
                        'cohen_kappa': make_scorer(cohen_kappa_score),
                        'rec': 'recall',
                        'prec': 'precision'
                    }
                    classifier = GridSearchCV(RandomForestClassifier(), parameters, scoring = scoring, refit="cohen_kappa")

            classifier.fit(data["features"], data["labels"].ravel())
            log_param("dataset", "training")

            if args.randomforest is not None and args.sk_gridsearch_rf is not None:
                results_df = sum_ranks_of_cv_results(classifier.cv_results_)
                save_results_as_csv(results_df)

        prediction = classifier.predict(data["features"])
        
        evaluation_metrics = select_metrics_based_on_args(args.metrics)
        computed_metrics = compute_metrics(
            evaluation_metrics, data["labels"], prediction)
        
        print_input_file_name(args.input_file) # eg training set
        print_formatted_metrics(computed_metrics) # eg Accuracy: 0.908
        log_metrics(computed_metrics)
        # export the trained classifier if the user wants us to do so
        if args.export_file is not None and args.sk_gridsearch_rf is None:
            output_dict = {"classifier": classifier, "params": params}
            with open(args.export_file, 'wb') as f_out:
                pickle.dump(output_dict, f_out)


def sum_ranks_of_cv_results(cv_results_):

    results_df = pd.DataFrame(cv_results_)
    results_df["rank_sum"] = (
        results_df["rank_test_cohen_kappa"] 
        + results_df["rank_test_rec"] 
        + results_df["rank_test_prec"])

    results_df.sort_values(by=['rank_sum'], inplace=True)

    return results_df


def save_results_as_csv(results_df):
    drop_cols = [
        "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time", 
        "params",
        "split0_test_cohen_kappa", "split1_test_cohen_kappa", "split2_test_cohen_kappa", 
        "split3_test_cohen_kappa", "split4_test_cohen_kappa", "std_test_cohen_kappa", 
        "split0_test_rec", "split1_test_rec", "split2_test_rec", "split3_test_rec", 
        "split4_test_rec", "std_test_rec", 
        "split0_test_prec", "split1_test_prec", "split2_test_prec",
        "split3_test_prec", "split4_test_prec", "std_test_prec"
    ]
    results_df.drop(columns=drop_cols, inplace=True)
    results_df.to_csv("data/gridsearch_results.csv", encoding="utf-8")


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


def parse_arguments():
    """
    parses the passed command line arguments to decide 
    which specific functionalities should be executed
    in this script
    """
 
    ap = argparse.ArgumentParser(description="Classifier")

    ap.add_argument(
        "input_file", help="path to the input pickle file")

    seed_msg = "seed for the random number generator"
    ap.add_argument(
        "-s", '--seed', type=int, help=seed_msg, default=None)

    export_msg = "export the trained classifier to the given location"
    ap.add_argument(
        "-e", "--export_file", help=export_msg, default=None)

    import_msg = "import a trained classifier from the given location"
    ap.add_argument(
        "-i", "--import_file", help=import_msg, default=None)

    ap.add_argument(
        "-d", '--dummyclassifier', choices=["most_frequent", "stratified"], default=None)

    knn_msg = "k nearest neighbor classifier with the specified value of k"
    ap.add_argument(
        "--knn", type=int, help=knn_msg, default=None)

    rf_msg = "Random Forest classifier with the specified number of estimators (trees)"
    ap.add_argument(
        "-r", "--randomforest", type=int, help=rf_msg, default=None)

    metric_msg = "Choose `none`, `all` or a specific metric for evaluation"
    metrics_choices = ["none", "all", METR_ACC,
                       METR_KAPPA, METR_PREC, METR_REC, METR_F1, METR_JAC]
    ap.add_argument(
        "-m", "--metrics", choices=metrics_choices, help=metric_msg,  default=METR_KAPPA)

    grid_msg = "Perform grid search on RandomForestClassifier. Param range is predifined!"
    ap.add_argument(
        "--sk_gridsearch_rf", action="store_true", help=grid_msg, default=None)

    log_msg = "where to log the mlflow results"
    default_path = "data/classification/mlflow"
    ap.add_argument(
        "--log_folder", help=log_msg, default=default_path)

    runname_msg = "sets the name of the run for logging purposes"
    ap.add_argument(
        "-n", "--run_name", help=runname_msg, default="")

    args = ap.parse_args()

    return args


if __name__ == "__main__":
    main()
