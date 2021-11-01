#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualizes some aspects of the data set.
To run this file you first need to run the preprocessing phase of the pipeline
"""

import csv
import pandas as pd
import matplotlib.pyplot as plt


def main():    
    df = pd.read_csv("data/preprocessing/labeled.csv",
                        quoting=csv.QUOTE_NONNUMERIC,
                        lineterminator="\n",
                        verbose=False,
                        dtype={"quote_url": object, "place": object, "tweet": object, "language": object, "thumbnail": object},
                        converters={'mentions': eval, 'photos': eval, 'urls': eval})

    plt.figure(figsize=(20,5))
    df.language.value_counts().plot(kind='bar')
    plt.title('Shows distribution of tweets per language')    
    plt.grid()
    plt.show()

    # lists number of tweets by language
    print(df["language"].value_counts())

    en_df = df[df["language"] == "en"]
    non_en_df = df[df["language"] != "en"]
    
    en_count = len(en_df["language"]) # 283240 english tweets
    non_en_count = len(non_en_df["language"])  # 12571 non english tweets
    non_en_percentage = non_en_count * 100 / en_count    
    print(non_en_percentage) # 4.438 percent
    

if __name__ == "__main__":
    main()