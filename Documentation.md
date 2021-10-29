## ZeitZeiton Group- MLinPractice

## Proprecessing 
This application is capable of handling correctly four languages, namely the four most frequent languages: English, Spanish, French and German. For the mentioned languages, we upload for each language and the corresponding lemmatizer and the list of the stopwords.

<img src="imgs/languages.png">

- first we tokenize the tweet into tokens in *code/tokenizer.py*, i.e, on a word-level. In addition, we get rid of the urls, mentions and tags, since we have these in separates columns.
- after the tokenization, we remove the punctuation within the tokens of the tweet in *code/punctuation_remover.py*
- after the deletion of the punctuation, we remove stopwords within the tokenised tweet with no punctuation. Therefore, we choose the corresponding stopword list depending on the language of the tweet. For now, we have the nltk's stopwords list of the following languages: German, English, Spanish and French. This happens in *code/stopwords_remover.py*
- the last step of the preprocessing is the lemmatization, we use lemmatization instead of stemming, because it is more accurate but we are aware of the price for this accuracy to pay is the time needed for the computations. It is not a big problem, since we run the preprocessing step only one time. The lemmitzation is implemented in *code/lemmatizer.py*.
# Documentation Example

Some introductory sentence(s). Data set and task are relatively fixed, so 
probably you don't have much to say about them (unless you modifed them).
If you haven't changed the application much, there's also not much to say about
that.
The following structure thus only covers preprocessing, feature extraction,
dimensionality reduction, classification, and evaluation.

## Evaluation

### Design Decisions

Which evaluation metrics did you use and why? 
Which baselines did you use and why?

### Results

How do the baselines perform with respect to the evaluation metrics?

### Interpretation

Is there anything we can learn from these results?

## Preprocessing

I'm following the "Design Decisions - Results - Interpretation" structure here,
but you can also just use one subheading per preprocessing step to organize
things (depending on what you do, that may be better structured).

### Design Decisions

Which kind of preprocessing steps did you implement? Why are they necessary
and/or useful down the road?

### Results

Maybe show a short example what your preprocessing does.

### Interpretation

Probably, no real interpretation possible, so feel free to leave this section out.

## Feature Extraction

Again, either structure among decision-result-interpretation or based on feature,
up to you.

### Design Decisions

Which features did you implement? What's their motivation and how are they computed?

### Results

Can you say something about how the feature values are distributed? Maybe show some plots?

### Interpretation

Can we already guess which features may be more useful than others?

## Dimensionality Reduction

If you didn't use any because you have only few features, just state that here.
In that case, you can nevertheless apply some dimensionality reduction in order
to analyze how helpful the individual features are during classification

### Design Decisions

Which dimensionality reduction technique(s) did you pick and why?

### Results

Which features were selected / created? Do you have any scores to report?

### Interpretation

Can we somehow make sense of the dimensionality reduction results?
Which features are the most important ones and why may that be the case?

## Classification

### Design Decisions

Which classifier(s) did you use? Which hyperparameter(s) (with their respective
candidate values) did you look at? What were your reasons for this?

### Results

The big finale begins: What are the evaluation results you obtained with your
classifiers in the different setups? Do you overfit or underfit? For the best
selected setup: How well does it generalize to the test set?

### Interpretation

Which hyperparameter settings are how important for the results?
How good are we? Can this be used in practice or are we still too bad?
Anything else we may have learned?
