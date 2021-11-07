# Documentation - [Patoali](https://trello.com/b/3pj6SkWa)

This document presents the author's work on the 'Machine Learning in Practice' project which took place during the summer term 2021 as a block seminar at Osnabrück University. The given task was to analyze a data set containing data science related tweets and predict with machine learning models whether a tweet will go viral or not. A tweet is defined as viral if it exceeds the arbitrary threshold of the sum of 50 likes and retweets. The data set _Data Science Tweets 2010-2021_ contains _data science_, _data analysis_ and _data visualizaion_ tweets from verified accounts on Twitter from 2010-2021. It was collected and [shared on kaggle.com](https://www.kaggle.com/ruchi798/data-science-tweets) by Ruchi Bhatia.

The lecturer Lucas Bechberger provided his students with a foundational codebase. The given codebase consists of multiple python (`.py`) and bash (`.sh`) scripts that resemble a basic pipeline of the processing steps _preprocessing_, _feature extraction_, _dimensionality reduction_ and _classification_ which is common for machine learning projects. The shell scripts can be used to run the whole pipeline or to run individual steps by invoking python scripts with specific command line arguments. Results of the pipeline steps are stored in `.pickle` files to reuse them in a separate application (`src\application\application.py`). The application offers a rudimentary Read–eval–print loop to predict the virality of the tweet a user inputs. The students task was to understand the code base and extend or replace given placeholder implementations with proper solutions to imrpove and measure the virality prediction.

## Evaluation

## Preprocessing

## Feature Extraction

## Dimensionality Reduction

## Classification
