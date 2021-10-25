# Documentation
Machine Learning in Practice block seminar, winter term 2021/22 @ [UOS](https://www.uni-osnabrueck.de/startseite/).  
Held by Lucas Bechberger, M.Sc.  
Group members: Dennis Hesenkamp, Iolanta Martirosov, Yannik Ullrich

---
### Table of contents
1. [Introduction](#introduction)
1. [Preprocessing](#preprocessing)
2. [Feature Extraction](#feature_extraction)
3. [Classification](#classification)
4. [Evaluation](#evaluation)

---

<!-- Introduction section -->
<a name='introduction'></a>

## Introduction

This document contains the documentation for our project, which aims to classify tweets as viral/non-viral based on multiple features derived from  

- the meta data of the tweet and
- the natural language features of the tweet.

The data set used is Ruchi Bhatia's [Data Science Tweets 2010-2021](https://www.kaggle.com/ruchi798/data-science-tweets) from [Kaggle](https://www.kaggle.com/). The code base on which we built our machine learning pipeline was provided by Lucas Bechberger (lecturer) and can be found [here](https://github.com/lbechberger/MLinPractice).


<!-- Preprocessing section -->
<a name='preprocessing'></a>

## Preprocessing

The data set provides the raw tweet as it has been posted as well as multiple features related to the tweet, for instance the person who published it, the time it has been published at, whether it contained any media (be it photo, video, url, etc.), and many more. We employed multiple preprocessing steps to transform the input data into a more usable format for feature extraction steps later on.

### Tokenization
In the lecture, Lucas implemented a tokenizer to disassemble tweets into individual words using the `nltk` library[^nltk]. This is done to split up the raw tweet into its constituents, i.e. the single words and punctuation signs it contains. By doing so, further processing and feature extraction can be performed by looking at the single components of a sentence/tweet as opposed to working with one long string.

### Stop word removal

### Punctuation removal
A feature for removing punctuation from the raw tweet has already been implemented by @lbechberger.

### Lemmatization
Lemmatization modifies an inflected or variant form of a word into its lemma or dictionary form. 
Through lemmatization, we can make sure that words - on a sematical level - get interpreted in the same way, 
even when inflected: 'walk' and 'walking', for example, stem from the same word and ultimately have the same meaning. 
Lemmatization, as opposed to stemming, which is computationally more effective, tries to take context into account, 
which is why we chose to implement it instead of stemming.

<!-- Feature extraction section -->
<a name='feature_extraction'></a>

## Feature Extraction

### Character length
Do shorter or longer tweets potentially go more viral? Character length extraction has been implemented as a first 
feature by @lbechberger.

### Month
Does the month in which the tweet was published have an impact on its virality? Are there times of the year in which 
the potential to go viral is higher, e.g. holiday season? Using the `datetime` module, we extract the month in which a 
tweet was published from the metadata.

### Sentiment analysis
Using the VADER (Valence Aware Dictionary and sEntiment Reasoner) framework ([PyPI](https://pypi.org/project/vaderSentiment/)) 
or [homepage](https://github.com/cjhutto/vaderSentiment )), we extract the sentiment of a tweet. VADER was built 
for social media and takes into account, among other factors, emojis, punctuation, and caps. The `polarity_score()` function 
returns a value for positive, negative, and neutral polarity, as well as an additional compound value with -1 representing 
the most negative and +1 the most positive sentiment. The classifier does not need training as it is pre-trained, 
unknown words, however, are simply classified as neutral.

<!-- Classifier section -->
<a name='classification'></a>

## Classification

<!-- Evaluation section -->
<a name='evaluation'></a>

## Evaluation

### Cohen's kappa
Robust against class imbalance



<!-- Footnotes -->
[^nltk]: <https://www.nltk.org/>

<!-- -->
