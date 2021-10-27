#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reads in the original csv files and creates labels for the data points.
Stores the result as a single pandas DataFrame in a pickle file.

Created on Tue Sep 28 15:55:44 2021

@author: lbechberger
"""

import os, argparse, csv
import pandas as pd
from src.util import COLUMN_LIKES, COLUMN_RETWEETS, COLUMN_PHOTOS, COLUMN_VIDEO, COLUMN_VIRAL, COLUMN_MEDIA

# setting up CLI
parser = argparse.ArgumentParser(description = "Creation of Labels")
parser.add_argument("data_directory", help = "directory where the original csv files reside")
parser.add_argument("output_file", help = "path to the output csv file")
parser.add_argument("-l", '--likes_weight', type = int, help = "weight of likes", default = 1)
parser.add_argument("-r", '--retweet_weight', type = int, help = "weight of retweets", default = 1)
parser.add_argument("-t", '--threshold', type = int, help = "threshold to surpass for positive class", default = 50)
parser.add_argument("-m", '--mediafile', type = str, choices = ['photo', 'video', 'both', 'none'],
                    help = "which media file to look at, 'photo', 'video, 'both' or 'none'", default = "both")
args = parser.parse_args()

# get all csv files in data_directory
file_paths = [args.data_directory + f for f in os.listdir(args.data_directory) if f.endswith(".csv")]

# load all csv files
dfs = []
for file_path in file_paths:
    dfs.append(pd.read_csv(file_path, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n"))

# join all data into a single DataFrame
df = pd.concat(dfs)

# compute new column "viral" based on likes and retweets
df[COLUMN_VIRAL] = (args.likes_weight * df[COLUMN_LIKES] + args.retweet_weight * df[COLUMN_RETWEETS]) > args.threshold

# adds new column based on video and photo existence
df[COLUMN_MEDIA] = "None"
if args.mediafile == "video" or args.mediafile == "both":
    df[COLUMN_MEDIA].mask(df[COLUMN_VIDEO] == 1, other="Video", inplace=True)
    df[COLUMN_MEDIA].mask(df[COLUMN_PHOTOS] != "[]", other="None", inplace=True)
if args.mediafile == "photo" or args.mediafile == "both":
    df[COLUMN_MEDIA].mask(df[COLUMN_PHOTOS] != "[]", other="Photo", inplace=True)


# print statistics
print("Number of tweets: {0}".format(len(df)))
print("Viral distribution:")
print(df[COLUMN_VIRAL].value_counts(normalize = True))
print("Media distribution for " + args.mediafile)
print(df[COLUMN_MEDIA].value_counts(normalize = True))

# store the DataFrame into a csv file
df.to_csv(args.output_file, index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")