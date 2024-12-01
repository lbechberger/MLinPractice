#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examples for different kinds of dimensionality reduction
"""

from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np

data_set = load_breast_cancer()
X = data_set.data
y = data_set.target
print("Data Set: ", X.shape, y.shape)
print("Combinatorics of binary feature values:", 2**30)


# PCA
print("\nPCA")
print('---')
pca = PCA(random_state = 42)
pca.fit(X)
print("explained variance (percentage): ", pca.explained_variance_ratio_)
print('most important component: ', pca.components_[0])
pca_transformed = pca.transform(X)
pca_transformed = pca_transformed[:,0:1]
print("after transformation: ", pca_transformed.shape, y.shape)
print("Compare: ", X[0], pca_transformed[0])


# wrapper
print("\nWrapper")
print("-------")

model = LogisticRegression(random_state = 42, max_iter = 10000)
rfe = RFE(model, n_features_to_select = 2)
rfe.fit(X,y)
print("Feature ranking according to RFE/LogReg:", rfe.ranking_)
index_of_first = np.where(rfe.ranking_ == 1)[0][0]
index_of_second = np.where(rfe.ranking_ == 2)[0][0]
print("Two most promising features: ", index_of_first, index_of_second)
wrapper_transformed = rfe.transform(X)
# or alternatively this does the same??
# wrapper_transformed = features[:,[index_of_first,index_of_second]]
# the line was in the preparatory dimensionalityReduction branch of lbechberger
print("After transformation: ", wrapper_transformed.shape, y.shape)
print("compare: ", X[0], wrapper_transformed[0])


# Filter
print("\n Filter")
print("------")
# mutual information (related to entropy and information gain when comparing data)
skb = SelectKBest(score_func = mutual_info_classif, k = 3)
skb.fit(X,y)
print("Feature scores according to MI: ", skb.scores_)
filter_transformed = skb.transform(X)
print("After transformation: ", filter_transformed.shape, y.shape)
print("Compare: ", X[0], filter_transformed[0])


# Embedded
print("\nEmbedded")
print("--------")
rf = RandomForestClassifier(n_estimators = 10, random_state=42)
rf.fit(X,y)
print("Feature importance according to RF: ", rf.feature_importances_)
sfm = SelectFromModel(rf, threshold = 0.1, prefit = True)
embedded_transformed = sfm.transform(X)
print("After transformation: ", embedded_transformed.shape, y.shape)
print("Compare: ", X[0], embedded_transformed[0])
