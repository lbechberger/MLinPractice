# Documentation
ML in Practice seminar, winter term 2021/22 @UOS  
Held by Lucas Bechberger, M.Sc.  
Group members: Dennis Hesenkamp, Iolanta Martirosov, Yannik Ullrich

This document contains the documentation for our project.

<!-- Classifier section -->
## Classifiers

<!-- Evaluation section -->
## Evaluation

### Cohen's kappa
Robust against class imbalance

<!-- Preprocessing section -->
## Preprocessing

### Tokenizer
In the lecture, we implemented a tokenizer to disassemble tweets into individual words. 
As proposed by @lbechberger, this was done using the nltk library.

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
## Feature extraction

### Character length
Do shorter or longer tweets potentially go more viral? Character length extraction has been implemented as a first 
feature by @lbechberger.

### Month
Does the month in which the tweet was published have an impact on its virality? Are there times of the year in which 
the potential to go viral is higher, e.g. holiday season? Using the `datetime` module, we extract the month in which a 
tweet was published from the metadata.

### Sentiment analysis
Using the VADER ((Valence Aware Dictionary and sEntiment Reasoner) framework ([PyPI](https://pypi.org/project/vaderSentiment/)) 
or [homepage](https://github.com/cjhutto/vaderSentiment )), we extract the sentiment of a tweet. VADER was built 
for social media and takes into account, among other factors, emojis, punctuation, and caps. The `polarity_score()` function 
returns a value for positive, negative, and neutral polarity, as well as an additional compound value with -1 representing 
the most negative and +1 the most positive sentiment. The classifier does not need training as it is pre-trained, 
unknown words, however, are simply classified as neutral.