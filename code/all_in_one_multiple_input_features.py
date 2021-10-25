import argparse

# import pdb
import csv
import pickle


# feature_extraction
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)

# feature_selection
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    chi2,
    mutual_info_regression,
)

# dim_reduction
from sklearn.decomposition import PCA, TruncatedSVD, NMF

# classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, l1_min_c, SVC, LinearSVR, SVR

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer

# metrics
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    accuracy_score,
    balanced_accuracy_score,
)

from code.util import (
    TEST_SIZE,
    HASH_VECTOR_N_FEATURES,
    NGRAM_RANGE,
    KNN_K,
    MAX_ITER_LOGISTIC,
    MAX_ITER_LINEAR_SVC,
    ALPHA_SGD,
    MAX_ITER_SGD,
)
import pandas as pd
import numpy as np

"""
# balancing
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
"""
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description="all in one")
parser.add_argument("input_file", help="path to the input file")
parser.add_argument(
    "-e",
    "--export_file",
    help="export the trained classifier to the given location",
    default=None,
)

# evaluate:
parser.add_argument(
    "-a", "--accuracy", action="store_true", help="evaluate using accuracy"
)
parser.add_argument(
    "-k", "--kappa", action="store_true", help="evaluate using Cohen's kappa"
)
parser.add_argument(
    "--balanced_accuracy", action="store_true", help="evaluate using balanced_accuracy"
)
parser.add_argument(
    "--classification_report",
    action="store_true",
    help="evaluate using classification_report",
)

# balance dataset
parser.add_argument(
    "--balance", type=str, help="choose btw under and oversampling", default=None
)
parser.add_argument("--small", type=int, help="choose subset of all data", default=None)
# feature_extraction
parser.add_argument(
    "--feature_extraction",
    type=str,
    help="choose a feature_extraction algo",
    default=None,
)
# dim_red
parser.add_argument("--dim_red", type=str, help="choose a dim_red algo", default=None)
# classifier
parser.add_argument("--classifier", type=str, help="choose a classifier", default=None)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="print information during training",
)
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")

if args.small is not None:
    # if limit is given
    max_length = len(df["label"])
    limit = min(args.small, max_length)
    df = df.head(limit)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    df, df["label"], test_size=TEST_SIZE, random_state=42
)

# print information during training
verbose = True if args.verbose else False

# use this code if you want to balance your dataset via under or over sampling
# you may have to modify it for your custom dataset.
# We don't need it here, because sklearn can do it in the classifier directly.
"""
# balance data
if args.balance == 'over_sampler':
    over_sampler = RandomOverSampler(random_state=42)
    X_res, y_res = over_sampler.fit_resample(X_train, y_train)
elif args.balance == 'under_sampler':
    under_sampler = RandomUnderSampler(random_state=42)
    X_res, y_res = under_sampler.fit_resample(X_train, y_train)
else:
    X_res, y_res = X_train, y_train


print(f"Training target statistics: {Counter(y_res)}")
print(f"Testing target statistics: {Counter(y_test)}")
"""


my_pipeline = []

# write functions for each column extraction to call it later in the pipeline
def get_text_data(x):
    return x["preprocess_col"]


def get_numeric_data(x):
    return x["video"].values.reshape(-1, 1)


# calculate the length of a tweet
def get_char_len(x):
    return x["tweet"].str.len().values.reshape(-1, 1)


# exists a photo in a tweet?
def get_photo_bool(x):
    return (x["photos"].str.len() > 2).values.reshape(-1, 1)


# at which hour was the post?
def get_hour(x):
    return pd.to_datetime(x["time"], format="%H:%M:%S").dt.hour.values.reshape(-1, 1)


# add text data if use just single feature
if args.feature_extraction != "union":
    my_pipeline.append(("selector", get_text_data))


# feature_extraction
if args.feature_extraction == "HashingVectorizer":
    my_pipeline.append(
        (
            "hashvec",
            HashingVectorizer(
                n_features=HASH_VECTOR_N_FEATURES,
                strip_accents="ascii",
                stop_words="english",
                ngram_range=NGRAM_RANGE,
            ),
        )
    )
elif args.feature_extraction == "TfidfVectorizer":
    my_pipeline.append(
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=NGRAM_RANGE))
    )

elif args.feature_extraction == "union":
    # using more than just text data as features
    my_pipeline.append(
        (
            "features",
            FeatureUnion(
                [
                    ("selector_numeric_data", FunctionTransformer(get_numeric_data)),
                    ("selector_char_len", FunctionTransformer(get_char_len)),
                    ("photo_bool", FunctionTransformer(get_photo_bool)),
                    ("select_hour", FunctionTransformer(get_hour)),
                    (
                        "text_features",
                        Pipeline(
                            [
                                ("selector_text", FunctionTransformer(get_text_data)),
                                (
                                    "vec",
                                    HashingVectorizer(
                                        n_features=HASH_VECTOR_N_FEATURES,
                                        strip_accents="ascii",
                                        stop_words="english",
                                        ngram_range=NGRAM_RANGE,
                                    ),  # change this to TfidfVectorizer if you want
                                ),
                            ]
                        ),
                    ),
                ],
                verbose=verbose,
            ),
        )
    )


# dimension reduction - not recommended here
if args.dim_red == "SelectKBest(chi2)":
    my_pipeline.append(("dim_red", SelectKBest(chi2)))
elif args.dim_red == "SelectKBest(mutual_info_regression)":
    my_pipeline.append(("dim_red", SelectKBest(mutual_info_regression)))
elif args.dim_red == "NMF":
    my_pipeline.append(("nmf", NMF()))


# classifier
if args.classifier == "MultinomialNB":
    # just use this without negative features
    my_pipeline.append(("MNB", MultinomialNB()))
elif args.classifier == "SGDClassifier":
    my_pipeline.append(
        (
            "SGD",
            SGDClassifier(
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
                verbose=verbose,
                alpha=ALPHA_SGD,
                max_iter=MAX_ITER_SGD,
            ),
        )
    )
elif args.classifier == "LogisticRegression":
    my_pipeline.append(
        (
            "LogisticRegression",
            LogisticRegression(
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
                verbose=verbose,
                max_iter=MAX_ITER_LOGISTIC,
            ),
        )
    )
elif args.classifier == "LinearSVC":
    my_pipeline.append(
        (
            "LinearSVC",
            LinearSVC(
                class_weight="balanced",
                random_state=42,
                verbose=verbose,
                max_iter=MAX_ITER_LINEAR_SVC,
            ),
        )
    )

elif args.classifier == "SVC":
    # attention: time = samples ^ 2
    my_pipeline.append(
        ("SVC", SVC(class_weight="balanced", random_state=42, verbose=verbose))
    )

classifier = Pipeline(my_pipeline)

# start training
classifier.fit(X_train, y_train)


# now predict the given data
prediction = classifier.predict(X_test)

prediction_train_set = classifier.predict(X_train)


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

    print("    {0}: {1}".format(metric_name, metric(y_test, prediction)))

# report table with recall, precision, and f1 score
if args.classification_report:
    categories = ["Flop", "Viral"]
    print("Matrix Train set:")
    print(classification_report(y_train, prediction_train_set, target_names=categories))
    print("Matrix Test set:")
    print(classification_report(y_test, prediction, target_names=categories))


# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    with open(args.export_file, "wb") as f_out:
        pickle.dump(classifier, f_out)
