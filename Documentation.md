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

- the metadata of the tweet and
- the natural language features of the tweet.

The data set used is Ruchi Bhatia's [Data Science Tweets 2010-2021](https://www.kaggle.com/ruchi798/data-science-tweets) from [Kaggle](https://www.kaggle.com/). The code base on which we built our machine learning pipeline was provided by Lucas Bechberger (lecturer) and can be found [here](https://github.com/lbechberger/MLinPractice). 

<p style='color:red'><b>On which basis have the labels in the data set been assigned?</b></p>


<!-- Preprocessing section -->
<a name='preprocessing'></a>

## Preprocessing

The data set provides the raw tweet as it has been posted as well as multiple features related to the tweet, for instance the person who published it, the time it has been published at, whether it contained any media (be it photo, video, url, etc.), and many more. We employed multiple preprocessing steps to transform the input data into a more usable format for feature extraction steps later on.

### Tokenization
In the lecture, Lucas implemented a tokenizer to disassemble tweets into individual words using the `nltk` library[^nltk]. This is done to split up the raw tweet into its constituents, i.e. the single words and punctuation signs it contains. By doing so, further processing and feature extraction can be performed by looking at the single components of a sentence/tweet as opposed to working with one long string.

Example:  

```python
>>> import nltk

>>> sent = 'There is great genius behind all this.'
>>> nltk.word_tokenize(sent)
['There', 'is', 'great', 'genius', 'behind', 'all', 'this', '.']
```


### Stopword removal
To extract meaningful natural language features from a string, it makes sense to first remove any stopwords occuring in that string. Say, for example, one would like to look at the most frequently occuring words in a large corpus. Usually, that means looking at words which actually carry _meaning_ in the given context. According to the OEC[^oec], the largest 21<sup>st</sup>-century English text corpus, the commonest word in English is _the_ - from which we cannot derive any meaning. Hence, it would make sense to remove words such as _the_ and other, non-meaning carrying words (= stopwords) from a corpus (the set of tweets in our case) before doing anything like keyword of occurence frequency analysis.  

There is not one universal stopword list nor are there universal rules on how stopwords should be defined. For the sake of convenience, we decided to use `gensim`'s `gensim.parsing.preprocessing.remove_stopwords` function[^gensim_stopwords], which uses `gensim`'s built-in stopword list containing high-frequency words with little lexical content.  

Example:  

```python
>>> import gensim

>>> sent = 'There is great genius behind all this.'
>>> gensim.parsing.preprocessing.remove_stopwords(sent)
'There great genius this.'
```

Other options would have been `nltk`'s stopword corpus[^nltk_stopwords], an annotated corpus with 2.400 stopwords from 11 languages or `spaCy`'s stopword list[^spacy_stopwords], but we faced problems implementing the former one while `gensim`'s corpus apparently contains more words and leads to better results compared to the latter.

### Punctuation Removal
Punctuation removal follows the same rationale as stopword removal: A dot, hyphen, or exclamation mark will probably occur often in the corpus, but without carrying much meaning at first sight (we can actually also infer features from punctuation, more about that in [Sentiment Analysis](#sentiment_analysis)). A feature for removing punctuation from the raw tweet has already been implemented by Lucas during the lecture using the `string` package. Again, alternatives can be used - for example with `gensim`, which offers a function for punctuation removal[^gensim-punctuation]. We decided not to change anything here, as the implemented method worked fine (and there is not much benefit in looking at a different list of punctuation signs anyways, as opposed to stopword lists, which can vary quite a lot).  

Example:

```python
>>> import string

>>> sent = "O, that my tongue were in the thunder's mouth!"
>>> punctuation = '[{}]'.format(string.punctuation)
>>> sent.replace(punctuation, '')
"O that my tongue were in the thunders mouth"
```
Caveat: the above code will actually not produce the desired output, but works in our implementation due to the different format of the input (we pass a `dtype object` as input). This is just to illustrate how our code and punctuation removal in general work.

### Lemmatization
Lemmatization modifies an inflected or variant form of a word into its lemma or dictionary form. Through lemmatization, we can make sure that words - on a semantical level - get interpreted in the same way, even when inflected: _walk_ and _walking_, for example, stem from the same word and ultimately carry the same meaning. Lemmatization, as opposed to stemming, which is computationally more effective, tries to take context into account, which is why we chose to implement it instead of stemming.


gensim.parsing.preprocessing.stem
https://radimrehurek.com/gensim/parsing/preprocessing.html#gensim.parsing.preprocessing.stem

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

<a name='sentiment_analysis'></a>
### Sentiment Analysis
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


## Conclusion
Different reserach questions:  
<p style='color:red'>How does tweet metadata play into virality?</p>



<!-- Footnotes -->
[^nltk]: <https://www.nltk.org/>
[^oec]: <https://web.archive.org/web/20111226085859/http://oxforddictionaries.com/words/the-oec-facts-about-the-language>, retrieved Oct 26, 2021
[^nltk_stopwords]: <https://www.nltk.org/book/ch02.html>, retrieved Oct 26, 2021
[^gensim_stopwords]: <https://radimrehurek.com/gensim/parsing/preprocessing.html#gensim.parsing.preprocessing.remove_stopwords>, retireved Oct 26, 2021
[^spacy_stopwords]: <https://github.com/explosion/spaCy/blob/master/spacy/lang/en/stop_words.py>, retrieved Oct 26, 2021
[^gensim-punctuation]: <https://radimrehurek.com/gensim/parsing/preprocessing.html#gensim.parsing.preprocessing.strip_punctuation>, retrieved Oct 26, 2021

<!-- -->
