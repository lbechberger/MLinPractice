import pdb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score, balanced_accuracy_score
import argparse
import csv
import pickle
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(description="Classifier")
parser.add_argument("input_file", help="path to the input pickle file")
args = parser.parse_args()


df = pd.read_csv(args.input_file, quoting=csv.QUOTE_NONNUMERIC,
                 lineterminator="\n")
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',
    categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)


text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None, verbose=True))])

text_clf.fit(twenty_train.data, twenty_train.target)


docs_test = twenty_test.data

predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))
print(classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
print(balanced_accuracy_score(predicted, twenty_test.target))
pdb.set_trace()
print(cohen_kappa_score(predicted, twenty_test.target))