#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example: exploratory data visualizuation with pandas and matplotlib
"""

# plotting with pandas
import csv
import pandas as pd

df = pd.read_csv("data/preprocessing/preprocessed.csv", quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

df["language"].value_counts().plot(kind = 'bar')
df["language"].value_counts().plot(kind = 'bar', logy = True)

df["date"] = df["date"].astype("datetime64")
df["label"].groupby(df["date"].dt.month).count().plot(kind = 'bar')


# plotting with matplotlib
import pickle
from matplotlib import pyplot as plt
import numpy as np

with open("data/feature_extraction/training.pickle", "rb") as f_in:
    data = pickle.load(f_in)

features = data["features"]
labels = data["labels"]

plt.hist(features)
plt.hist(features, range = [0,400])

pos = features[labels]
neg_index = np.array([not x for x in labels])
neg = features[neg_index]

bins = [0, 50, 100, 150, 200, 250, 300, 350, 400]

plt.hist(pos, bins = bins)
plt.hist(neg, bins = bins)
