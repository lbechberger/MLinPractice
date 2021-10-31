#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produces some of the plots used in the documentation

Created on Sun Oct 31 23:32:44 2021

@author: dhesenkamp
"""

import numpy as np
import pandas as pd
import time
import csv
import matplotlib.pyplot as plt


# courtesy to https://www.kaggle.com/docxian/data-science-tweets for some code snippets
# load data
df = pd.read_csv('data/raw/data_science.csv', quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

# convert date
df.date = pd.to_datetime(df.date)
# and extract year, month
df['year'] = df.date.dt.year
df['month'] = df.date.dt.month

# year distribution
df.year.value_counts().sort_index().plot(kind='bar', color='#b84064')
plt.title('Year of Tweet')

plt.savefig('img/year_distribution.png')
plt.show()

# language distribution
plt.figure(figsize=(20,6))
df.language.value_counts().plot(kind='bar', color="#b84064")
plt.title('Language')
#plt.grid()
plt.savefig('img/lang_distribution.png')
plt.show()

# time range distribution
# initializing data
preprocessed = pd.read_csv("data/preprocessing/preprocessed.csv", quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

daytime = []
day = preprocessed["time"]
for i in day:
    t = i.split(":")
    hour = int(t[0])
    
    # night hours
    if hour in range(0, 5):
        daytime.append(0)
    
    # morning hours
    if hour in range(5, 10):
        daytime.append(1)
    
    # midday
    if hour in range(10, 15):
        daytime.append(2)
    
    # afternoon
    if hour in range(15, 19):
        daytime.append(3)
    
    # evening hours
    if hour in range(19, 24):
        daytime.append(4)

d = np.zeros(5)
for i in daytime:
    if i == 0:
        d[0]+=1
    elif i == 1:
        d[1]+=1
    elif i == 2:
        d[2]+=1
    elif i == 3:
        d[3]+=1
    else:
        d[4]+=1

plt.title("Tweets per time range.")
plt.ylabel("Number of tweets")
plt.xlabel("Daytime")
r = ["night","morning","midday","afternoon","evening"]
plt.bar(r, d, align = "center", color="#b84064")
plt.savefig('img/time_distribution.png')

# final results
x = ['accuracy', 'balanced\naccuracy', 'cohens\nkappa', 'f1 score']
results = [0.63, 0.606, 0.088, 0.223]
plt.bar(x, results, 0.5, color='#b84064')
plt.title('CNB classifier on test set.')
plt.ylabel("Score")
plt.xlabel("Metric")
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig('img/final_result.png')
