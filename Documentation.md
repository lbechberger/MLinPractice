# Documentation Example

This is the forked repository for Magnus Müller, Maximilian Kalcher and Samuel Hagemann. 

Our task involved building and documenting a real-life application of machine learning. 
We were given a dataset of N tweets from the years X until Y and had to build a classifier that would detect whether a tweet would go viral. 
The measure for it being viral was when the sum of likes and retweets were bigger than 50. 

The dataset was very variable and we had a lot of features to work with, which gave us the freedom to choose and experiment with these freely. 

At the end, our classifier is implemented into an 'application', callable by terminal, which gives the likeliness of an input tweet being viral, having used the dataset as training. 

//Some introductory sentence(s). Data set and task are relatively fixed, so 
probably you don't have much to say about them (unless you modifed them).
If you haven't changed the application much, there's also not much to say about
that.
The following structure thus only covers preprocessing, feature extraction,
dimensionality reduction, classification, and evaluation.

## Preprocessing

Before using the data or some aspects of it, it is important to process some of it beforehand so our chosen features can be extracted smoothly. 
Many tweets had different kind of punctuation, ..., emojis, and some of them even were written in different languages.

### Design Decisions

After looking at the dataset closely, we chose to keep the core words of the sentence, ...

### Results

Maybe show a short example what your preprocessing does.

### Interpretation

Probably, no real interpretation possible, so feel free to leave this section out.

## Evaluation

### Design Decisions

Which evaluation metrics did you use and why? 
Which baselines did you use and why?

### Results

How do the baselines perform with respect to the evaluation metrics?

### Interpretation

Is there anything we can learn from these results?

## Feature Extraction

Again, either structure under decision-result interpretation or based on features,
is up to you.



### Design Decisions

Which features did you implement? What's their motivation and how are they computed?

We want to try something we didn't hear in the lecture. Therefore, we used the HashingVectorizer from sklearn to create an individual hash for each tweet. For a sentence like 'I love Machine Learning', the output can look like [0.4, 0.3, 0.9, 0, 0.21], with length n representing the number of features. It's not very intuitive to humans why this works, but after a long time of version conflicts and other problems, we enjoyed the simplicity of using sklearn. 

### Results

Can you say something about how the feature values are distributed? Maybe show some plots?

When we finally ran it successfully with 25 features, we tried it with the SVM classifier, but that took too much time (nearly endless), so we used KNN with 4 NN on a 20000 sample subset and for the first time our Cohen kappa went from 0.0 to 0.1 and after some tuning (using more data) to 0.3.


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
First of all we add a new argument: --small 1000 which would just use 1000s tweets.
### Design Decisions

Which classifier(s) did you use? Which hyperparameter(s) (with their respective
candidate values) did you look at? What were your reasons for this?

- SVM
### Results

The big finale begins: What are the evaluation results you obtained with your
classifiers in the different setups? Do you overfit or underfit? For the best
selected setup: How well does it generalize to the test set?

### Interpretation

Which hyperparameter settings are how important for the results?
How good are we? Can this be used in practice or are we still too bad?
Anything else we may have learned?
