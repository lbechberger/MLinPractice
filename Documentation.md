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