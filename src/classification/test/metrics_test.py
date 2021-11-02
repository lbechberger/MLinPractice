#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests the calculation of metrics
"""

import unittest
import csv
import pandas as pd
from src.classification.run_classifier import compute_metrics, select_metrics_based_on_args
from src.util import fm
import os


class MetricsTest(unittest.TestCase):

    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        # loads a test data set that replicates the quantities
        # from the VIPs questions about evaluation metrics       
        # 
        # |                        | Actual class: True | Actual class: False |
        # |------------------------|--------------------|---------------------|
        # | predicted class: True  | 23 (TP)            | 38 (FP)             |
        # | predicted class: False | 13 (FN)            | 682 (TN)            |

        self.df = pd.read_csv(dir_path + "/metrics_test_data.csv",
                              quoting=csv.QUOTE_NONNUMERIC,
                              lineterminator="\n",
                              dtype={"label": bool, "prediction": bool},
                              verbose=True)

    def test_compute_metrics(self):
        evaluation_metrics = select_metrics_based_on_args("all")

        labels = self.df["actual"]
        prediction = self.df["prediction"]
        computed_metrics = compute_metrics(
            evaluation_metrics, labels, prediction)

        computed_metrics_dict = dict(computed_metrics)

        DECIMAL_PLACES = 4

        msg = fm("a specific data set",
                 "calculate exact value for accuracy")
        score = round(computed_metrics_dict["Accuracy"], DECIMAL_PLACES)
        self.assertEqual(score, 0.9325, msg)

        msg = fm("a specific data set",
                 "calculate exact value for Cohen_kappa")
        score = round(computed_metrics_dict["Cohen_kappa"], DECIMAL_PLACES)
        self.assertEqual(score, 0.4407, msg)

        msg = fm("a specific data set",
                 "calculate exact value for Precision")
        score = round(computed_metrics_dict["Precision"], DECIMAL_PLACES)
        self.assertEqual(score, 0.3770, msg)

        msg = fm("a specific data set",
            "calculate exact value for Recall")
        score = round(computed_metrics_dict["Recall"], DECIMAL_PLACES)
        self.assertEqual(score, 0.6389, msg)

        msg = fm("a specific data set",
            "calculate exact value for F1-Score")
        score = round(computed_metrics_dict["F1-Score"], DECIMAL_PLACES)
        self.assertEqual(score, 0.4742, msg)

        msg = fm("a specific data set",
            "calculate exact value for Jaccard")
        score = round(computed_metrics_dict["Jaccard"], DECIMAL_PLACES)
        self.assertEqual(score, 0.3108, msg)
