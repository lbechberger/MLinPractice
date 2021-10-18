import unittest
import pandas as pd
import numpy as np
import pdb
import argparse
from code.classification.run_classifier import load_dataset, create_classifier
from argparse import Namespace

# from app import process_data


class TestClassifier(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser(description="Classifier")

        self.small_len = 10
        self.args = Namespace(
            input_file="data/feature_extraction/training.pickle",
            small=self.small_len,
            seed=42,
            balanced_data_set=True,
            import_file=None,
            majority=False,
            frequency=False,
            sgd=False,
            svm=False,
            LinearSVC=False,
            MultinomialNB=False,
            LogisticRegression=False,
            knn=1,
        )
        self.data = load_dataset(self.args)
        self.all_clf = [
            "majority",
            "frequency",
            "sgd",
            "LogisticRegression",
            "MultinomialNB",
            "LinearSVC",
            "svm",
        ]

    def test_load_dataset(self):

        # small arg is working
        self.assertEqual(len(self.data["labels"]), self.small_len)

        # dict has 3 output keys
        self.assertEqual(len(self.data), 3)

        # is dict
        self.assertTrue(type(self.data), dict)

    def test_create_classifier(self):
        for clf in self.all_clf:
            args_dict = vars(self.args)
            args_dict[clf] = True
            try:
                classifier = create_classifier(self.args, self.data)
            except:
                raise Exception("{clf} was not created successfully.")

            args_dict[clf] = False
            pdb.set_trace()
    
    def test_no_classifier(self):
        self.args.knn = None
        with self.assertRaises(UnboundLocalError):
            create_classifier(self.args, self.data)


if __name__ == "__main__":
    unittest.main()
