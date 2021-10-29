# Documentation
Machine Learning in Practice block seminar, winter term 2021/22 @ [UOS](https://www.uni-osnabrueck.de/startseite/).  
Held by Lucas Bechberger, M.Sc.  
Group members: Dennis Hesenkamp, Yannik Ullrich

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
import nltk

sent = 'There is great genius behind all this.'
nltk.word_tokenize(sent)

# ['There', 'is', 'great', 'genius', 'behind', 'all', 'this', '.']
```


### Stopword Removal
To extract meaningful natural language features from a string, it makes sense to first remove any stopwords occuring in that string. Say, for example, one would like to look at the most frequently occuring words in a large corpus. Usually, that means looking at words which actually carry _meaning_ in the given context. According to the OEC[^oec], the largest 21<sup>st</sup>-century English text corpus, the commonest word in English is _the_ - from which we cannot derive any meaning. Hence, it would make sense to remove words such as _the_ and other, non-meaning carrying words (= stopwords) from a corpus (the set of tweets in our case) before doing anything like keyword of occurence frequency analysis.  

There is not one universal stopword list nor are there universal rules on how stopwords should be defined. For the sake of convenience, we decided to use `gensim`'s `gensim.parsing.preprocessing.remove_stopwords` function[^gensim_stopwords], which uses `gensim`'s built-in stopword list containing high-frequency words with little lexical content.  

Example:  

```python
import gensim

sent = 'There is great genius behind all this.'
gensim.parsing.preprocessing.remove_stopwords(sent)

# 'There great genius this.'
```

Other options would have been `nltk`'s stopword corpus[^nltk_stopwords], an annotated corpus with 2.400 stopwords from 11 languages or `spaCy`'s stopword list[^spacy_stopwords], but we faced problems implementing the former one while `gensim`'s corpus apparently contains more words and leads to better results compared to the latter.


### Punctuation Removal
Punctuation removal follows the same rationale as stopword removal: A dot, hyphen, or exclamation mark will probably occur often in the corpus, but without carrying much meaning at first sight (we can actually also infer features from punctuation, more about that in [Sentiment Analysis](#sentiment_analysis)). A feature for removing punctuation from the raw tweet has already been implemented by Lucas during the lecture using the `string` package. Again, alternatives can be used - for example with `gensim`, which offers a function for punctuation removal[^gensim-punctuation]. We had to rebuild this class as it was initially meant to work as first step in the preprocessing pipeline, but we now have it in second place. Hence, it was necessary to change how the class handles input and output and we needed an additional command line argument. We did not change the method of removing punctuation in general, as there is not much benefit in looking at different ways of punctuation removal anyways, as opposed to stopword removal, where lists can vary a lot based on the corpus.

Example:

```python
import string

sent = "O, that my tongue were in the thunder's mouth!"
punctuation = '[{}]'.format(string.punctuation)
sent.replace(punctuation, '')

# "O that my tongue were in the thunders mouth"
```
Caveat: the above code will actually not produce the desired output, but works in our implementation due to the different format of the input (we pass a `dtype object` as input). This is just to illustrate how our code and punctuation removal in general work.


### Lemmatization
Lemmatization modifies an inflected or variant form of a word into its lemma or dictionary form. Through lemmatization, we can make sure that words - on a semantical level - get interpreted in the same way, even when inflected: _walk_ and _walking_, for example, stem from the same word and ultimately carry the same meaning. We decided to use lemmatization as opposed to stemming, although it is computationally more expensive. This is due to lemmatization taking context into account, as it depends on part-of-speech (PoS) tagging.  

To implement this, we used `nltk`'s `pos_tag` to assign PoS tags and WordNet's `WordNetLemmatizer()` class, as well as a manually defined PoS dictionary to reduce the (rather detailed) tags from `pos_tag` to only four different, namely _noun_, _verb_, _adjective_, and _adverb_:

```python
from nltk.corpus import wordnet

tag_dict = {"J": wordnet.ADJ,
			"N": wordnet.NOUN,
			"V": wordnet.VERB,
			"R": wordnet.ADV}
```

This simplified PoS assignment is important because `pos_tag` returns a tuple, which has to be converted to a format the WordNet lemmatizer can work with, further WordNet lemmatizes differently for different PoS classes and only distinguishes between the above mentioned classes. Courtesy to [this blog entry](https://www.machinelearningplus.com/nlp/lemmatization-examples-python/#wordnetlemmatizerwithappropriatepostag) by Selva Prabhakaran for the idea and the code.

Example:

```python
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

sent = ['These', 'newer', 'data', 'help', 'scientists', 'accurately', 'project', 'how', 'quickly', 'glaciers', 'are', 'retreating', '.']
lem = WordNetLemmatizer()
lemmatized = []

for word in sent:
	# get the part-of-speech tag
	tag = pos_tag([word])[0][1][0].upper()
	lemmatized.append(lem.lemmatize(word.lower(), tag_dict.get(tag, wordnet.NOUN)))
	
# ['these', 'newer', 'data', 'help', 'scientist', 'accurately', 'project', 'how', 'quickly', 'glacier', 'be', 'retreat', '.']
```

Whenever the PoS tagging encounters an unknown tag or a tag which the lemmatizer cannot handle, the default tag to be used is `wordnet.NOUN`.

As mentioned in the beginning, alternatively to lemmatization we could use the computationally cheaper stemming, which only reduces an inflected word to its stem (e.g. _accurately_ becomes _accur_). This could be done with `gensim.parsing.preprocessing.stem`[^gensim_stemming]


### Final Words
The above preprocessing steps have all been tested and work fine. Some of them can be performed independently, but we built the pipeline such that they stack. To ensure proper functionality, the order of steps has to be as follows:

1. Stopword removal
2. Punctuation removal
3. Tokenization
4. Lemmatization

Input columns have to be specified accordingly with the provided command line arguments (see readme for more info).


<!-- Feature extraction section -->
<a name='feature_extraction'></a>

## Feature Extraction
After the preprocessing of the data is done, we can move on to extracting features from the dataset.


### Character Length
The length of a tweet might influence its chance of going viral as people might prefer shorter texts on social media (or longer, more complex ones). This feature was already implemented by Lucas as an example, using `len()`.

Example:

```python
sent = 'There is great genius behind all this.'
len(sent)

# 38
```

This is, however simple it may be, a difficult to interpret feature: for most of its existence, Twitter has had a character limit of 140 characters per tweet. In 2017, the maximum character limit was raised to 280[^twitter_charlength], which lead to an almost immediate drop of the prevalence of tweets with around 140 characters while, at the same time, tweets approaching 280 characters appear to be snytactically and semantically similar to tweets around 140 characters from before the change __(Gligoric, Anderson, West, 2020)__.

<a name='month'></a>
### Month
We thought that the month in which a tweet was published could have (some minor?) influence on its virality. Maybe during holiday season or the darker time of the year, i.e. from October to March, people spend more time on the internet, hence tweets might get more interaction which will lead to a higher potential of going viral.

We extracted the month from the `date` column of the dataframe using the `datetime` package as follows:

```python
import datetime

date = "2021-04-14"
datetime.datetime.strptime(date, "%Y-%m-%d").month

# 4
```

The result we return is the respective month. We have NOT yet implemented one-hot encoding for the result because we actually decided rather quickly that we do not want to use this feature. We could not find evidence or reserach on our assumption that screentime/time on the internet is higher during certain months or periods of the year. How one-hot encoding is done can be seen in [Time of Day](#time_of_day).


<a name='sentiment_analysis'></a>
### Sentiment Analysis
The tonation of 
Using the VADER (Valence Aware Dictionary and sEntiment Reasoner)[^vader_pypi] [^vader_homepage] framework, we extract the compound sentiment of a tweet. VADER was built for social media and takes into account, among other factors, emojis, punctuation, and caps - which is why we let it work on the unmodified `tweet` column of the dataframe, ensuring that we do not artificially modify the sentiment. The `polarity_score()` function returns a value for positive, negative, and neutral polarity, as well as an additional compound value with $-1$ representing the most negative and $+1$ the most positive sentiment. The classifier does not need training as it is pre-trained, unknown words, however, are simply classified as neutral. 

Example:

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
sentences = ["The service here is good.", 
			"The service here is extremely good.", 
			"The service here is extremely good!!!", 
			"The service here is extremely good!!! ;)"]

for s in sentences:
 	print(sia.polarity_scores(s)['compound'])
 	
# 0.4404
# 0.4927
# 0.6211
# 0.7389 	
```
We can see how the compound sentiment changes with the addition of words, punctuation, and emojis. We decided to only use the compound sentiment as measure because we felt that this is the most important one. A tweet might have a certain negativity score (indicating, e.g., that it is negatively phrased) because of a few words, while the rest of the tweet is phrased very positively, resulting in a positive compound sentiment. However, compared to a tweet with only neutral phrasing (i.e. a negative score of $0$), it would still be classified as more negative, which would intuitively be wrong.


<a name='time_of_day'></a>
### Time of Day
As opposed to the [Month](#month) feature, which we ended up not using, we felt that the time of the day during which a tweet was posted might very well have an influence on its virality. We decided to split the day in time ranges with hard boundaries:

1. Morning hours from 5am to 9am (5 hours)
2. Midday from 10am to 2pm (5 hours)
3. Afternoon from 3pm to 6pm (4 hours)
4. Evening from 7pm to 11pm (5 hours)
5. Night from 12am to 4am (5 hours)

The time sections are roughly equally sized, with afternoon being the only exception. We decided to split the day like this based on our own experience. We have not tested different splits. Alternatively, one could test 

1. Diffferent time ranges
2. A finer split, i.e. more categories
3. A less fine split, i.e. less categories

We extracted the time from the `time` column of the dataframe and simply used the `split()` method to extract the hour from the string it is stored in, then checked whether the extracted value falls into a predefined range and appened the respective value to a new column. We then one-hot encoded the result to retrieve a binary classification for every entry with `pandas`' `get_dummies()` function.

```python
import pandas as pd

time = ["05:15:01", "07:11:31", "16:04:59", "23:12:00"]
hours = [t.split(":")[0] for t in time]
# ['05', '07', '16', '23']

result = []
for h in hours:
	if hour in range(0, 6):
		result.append(0)
	elif hour in range(6, 11):
		result.append(1)  
	elif hour in range(11, 15):
		result.append(2)
	elif hour in range(15, 19):
		result.append(3)  
	elif hour in range(19, 24):
		result.append(4)
		
pd.get_dummies(result) 

#	05	07	16	23
#0	1	0	0	0
#1	0	1	0	0
#2	0	0	1	0
#3	0	0	0	1
# only yields encoding for 4 variables in this case because 5th category not used
```


### URLs, Photos, Mentions, Hashtags
In this section, we evaluate whether any of the above have been attached to a tweet as a binary (1 if attached, 0 else). Our thinking here was that additional media, be it a link, pictures, mentions of another account, or hashtags, might influence the potential virality of a tweet. We accessed the respective columns of the dataframe (`url`, `photos`, `mentions`, `hashtags`), in which the entries are stored in a list. Hence, we could simply evaluate the length of the entries. If they exceed a length of 2, the column contains more than just the empty brackets and the tweet contains the respective feature:

Example with URL:

```python
urls = ["[]", "['https://www.sciencenews.org/article/climate-thwaites-glacier-under-ice-shelf-risks-warm-water']", "[]"]

result = [0 if len(url) <= 2 else 1 for url in urls]

# [0, 1, 0]
```

Important: although being stored in lists, the column entries get still evaluated as strings. That is why checking for a length less equal 2 works in this case. The evaluation procedure (checking for the length) is the same for all of the above features.


### Replies
We also figured that the number of replies has an influence on the virality: the more people engage with a tweet and reply to it, the more people see it in their news feed, which again increases reach and interactions. The number of replies are stored as float in the column `replies_count` of the dataframe, so we just have to access that column, make a copy, transform it to a `numpy.array`, and reshape it so the classifier can work with the data later on:

```python
import numpy as np

replies = [0.0, 7.0, 2.0, 49.0]
np.array(replies).reshape(-1, 1)

# array([[ 0.],
#		[ 7.],
#		[ 2.],
#		[49.]])
```


### Retweets and Likes
Retweets and likes follow the same rationale as replies. These are the most obvious features to consider when measuring virality and we just implented them for the purpose of testing. We did not use them for training our model (since that easily results in an accuracy $>99\%$ and does not tell us anything about _why_ the tweet went viral). The procedure is the same as above: access the respective column, convert it to a `numpy.array` and reshape it.

<!-- Dimensionality reduction section -->
<a name='dimensionality_reduction'></a>

## Dimensionality Reduction


<!-- Classifier section -->
<a name='classification'></a>

## Classifier

### _k_ - Nearest Neighbour
```python
from sklearn.neighbors import KNeighborsClassifier
```
The _k_-NN classifier was implemented by Lucas during the lecture. We use it with only one hyperparameter - _k_ - for our binary classification task. This algorithm is an example for instance-based learning. It relies on the distance between data points for classification, hence it requires standardization of feature vectors.

We decided to additionally implement a way to change the weight function. As default, the `KNeighborsClassifier` works with uniform weights, i.e. all features are equally important. Having an additional option for distance-weighted classification where nearer neighbors are more important than those further away made sense for us (and it also improved our results, as can be seen later).

Other than that, though, we left the classifier with default settings. A notable alternative could have been the choice of the algorithm for computation of the nearest neighbors, options being a brute-force search, _k_-dimensional tree search, and ball tree search. The default option is `auto`, where the classifier picks the method it deems fittest for the task at hand.


<a name='decision_tree'></a>
### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier
```

Further, we implemented a decision tree classifier. Due to its nature of learning decision rules from the dataset, it does neither require standardization of data nor does it make assumptions on the data distribution.

We added the option to define a maximum depth of the tree, which is extremely important to cope with overfitting. Further, the criterion for measuring split quality can be choosen between Gini impurity and entropy/information gain. The former is usually preferred for classification and regression trees (CART) while the latter is used for the ID3 variant[^id3] of decision trees. Although `sklearn` employs a version of the CART algorithm, it nonetheless works with entropy as measure.

Decision trees generally have difficulties working with continuous data and we have the compound sentiment (see [Sentiment Analysis](#sentiment_analysis)) as feature of such nature which is continuous in the range $[-1, 1]$ (although a case could be made for it being a discrete feature since it is rounded to four decimal places).


### Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
```

Random forest classifiers represent an ensemble of multiple decision trees. They are often more robust and accurate than single decision trees and less prone to overfitting and can, therefore, better generalize on new, unseen data.

We implemented it such that we can modify the number of trees per forest, the maximum depth per tree, as well as the criterion based on which a split occurs. The options for this are the same as for single decision trees - Gini impurity and entropy. The first two are the main parameters to look at when constructing a forest according to `sklearn`'s user guide on classifiers[^sklearn_forest]. Usually, the classification is obtained via majority vote of the trees in a forest, but this implementation averages over the probabilistic class prediction of the single classifiers.

Being able to manipulate both the maximum depth as well as the split criterion further allows us to compare the forest to our single decision tree classifier, since we can use the same parametrization for both.


### Support Vector Machine
```python
from sklearn.svm import SVC
```

We also added a support vector machine. This classifier aims to maximise the decision boundaries between different classes.


<!-- Evaluation section -->
<a name='evaluation'></a>

## Evaluation Metrics
We implemented multiple evaluation metrics to see how well our classification works.

### Accuracy


### Balanced Accuracy


### Cohen's Kappa
Robust against class imbalance

### F<sub>1</sub> - Score



## Hyperparameter Optimization
knn: only odd k values

forest: large number of trees = high accuracy Usually


## Conclusion
Different reserach questions:  
<p style='color:red'>How does tweet metadata play into virality?</p>


## Resources
<https://towardsdatascience.com/comparative-study-on-classic-machine-learning-algorithms-24f9ff6ab222>


<!-- Footnotes -->
[^nltk]: <https://www.nltk.org/>
[^oec]: <https://web.archive.org/web/20111226085859/http://oxforddictionaries.com/words/the-oec-facts-about-the-language>, retrieved Oct 26, 2021
[^nltk_stopwords]: <https://www.nltk.org/book/ch02.html>, retrieved Oct 26, 2021
[^gensim_stopwords]: <https://radimrehurek.com/gensim/parsing/preprocessing.html#gensim.parsing.preprocessing.remove_stopwords>, retireved Oct 26, 2021
[^spacy_stopwords]: <https://github.com/explosion/spaCy/blob/master/spacy/lang/en/stop_words.py>, retrieved Oct 26, 2021
[^gensim-punctuation]: <https://radimrehurek.com/gensim/parsing/preprocessing.html#gensim.parsing.preprocessing.strip_punctuation>, retrieved Oct 26, 2021
[^gensim_stemming]: <https://radimrehurek.com/gensim/parsing/preprocessing.html#gensim.parsing.preprocessing.stem>, retrieved Oct 26, 2021
[^vader_pypi]: <https://pypi.org/project/vaderSentiment/>
[^vader_homepage]: <https://github.com/cjhutto/vaderSentiment>
[^twitter_charlength]: <https://blog.twitter.com/official/en_us/topics/product/2017/Giving-you-more-characters-to-express-yourself.html>
[^sklearn_forest]: <https://scikit-learn.org/stable/modules/ensemble.html#forest>
<!-- -->
