# Documentation - ML Pipeline - Team One 

## Research Hypothesis

The goal of our example project is to predict which tweets will go viral, i.e., receive many likes and retweets. This criterion is defined by the sum of likes and retweets, where the threshold is specifiable by the user, but defaults to 50.

## Data Collection

As data source, we use the "Data Science Tweets 2010-2021" data set (version 3) by Ruchi Bhatia from [Kaggle](https://www.kaggle.com/ruchi798/data-science-tweets). 

## Preprocessing

pipeline tweet replace_urls expand standardize lemmatize remove_stopwords

### Punctuation

#### Goal
We remove all punctuation from tweet texts in order to focus on word semantics in our classification and match words at different position within a sentence. This also removes hashtag-signs, conveniently allowing us to treat hashtags as normal words.

#### Implementation Process
We implemented a [PunctuationRemover class](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/punctuation_remover.py) which uses a string replacement function to remove punctuation.

#### Discussion
Removing punctuation may result in loss of context in cases where punctuation is used to convey emotional content (as with exclamation marks) or vital sentence meaning (as with question marks and commas). However, we believe that punctuation in the context of tweets only marginally influences meaning as many tweeters omit punctuation anyway and the character limit of 240 generally leads to less complex grammatical structure.  

### Lowercase

#### Goal
We lowercase all tweet texts in order to be able to reliably match different capitalizations of the same word in downstream preprocessors as well as the classifier.

#### Implementation Process
We implemented a [Lowercase class](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/lowercase.py) that uses python's lowercase funcationality.

#### Discussion
This step is generally useful for natural language processing and does not result in the loss of task specific features.

### Expand Contractions
#### Goal
Contractions add redundancy, as they technically are seperate tokens, eventhough their components overlap with other tokens. They also usually don't carry much semantic value, since their components are usually stop words. Expanding them to their respective long form removes this redundancy and assists the stopword removal process that occurs at a later point in the pipeline. 
Ex.: isn't --> is not --> 'is' 'not'
#### Implementation Process
The contractions are expanded by the [Expander](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/expand.py), while the contraction mapping can be found [here](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/util/contractions.py). The Implementation stems from [towardsdatascience.com](https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72) and uses a list mapping contractions to their respective long forms. 
#### Discussion
Adding this preprocessing step is not necessarily crucial to the preprocessing and one might argue that removing it may speed up the pipeline. However, it is a simple way to minimize the vocabulary of our dataset by avoiding unnecessary duplicate tokens and to ensure the fidelity of our model to semantics. To clarify, tokens with the same semantics should be classified as one item in a vocabulary, no matter if they are contracted or not.

### Standardize Spelling Variations
#### Goal
Spelling variations arise due to misspellings or location-based differences. Different spellings for the same word add redundancy to our features, as they are counted as different vocabulary, eventhough their semantics are the same. Changing variations of words, in our case location-base differences, to a standard spelling ensures that semantic information for the words is kept and that they can be further dealt with as the same word.
#### Implementation Process
The tweets are standardize by the [Standardizer](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/standardize.py) class, while the spellings mapping can be found [here](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/util/spellings.py). The Implementation is in line with the implementation for the expansion of contractions, as seen above, and uses as mapping of UK spellings to their respective US spellings. We treat the US spellings as the standard vocabulary, and change any UK variations to the standard US spelling.
#### Discussion
Adding this preprocessing step is not necessarily crucial to the preprocessing and one might argue that removing it may speed up the pipeline. However, it is a simple way to minimize the vocabulary of our dataset by avoiding unnecessary duplicate tokens and to ensure the fidelity of our model to semantics. To clarify, tokens with the same semantics should be classified as one item in a vocabulary, no matter if they are contracted or not.

### Tokenizer

#### Goal
To ease feature extraction from tweet texts we split them at word boundaries.
We split the tweet texts into lists by word boundaries in order to be able to count them and run statistical feature extraction more easily.

#### Implementation Process
The tweet texts are tokenized using [NLTK (Natural Langauge Toolkit)](https://www.nltk.org/) in the [Tokenizer class](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/tokenizer.py).

#### Discussion
This step is generally necessary to process natural langauge and aids in the classification of the tweets.

### Remove numbers
#### Goal
Number expressions have a high variance without much meaningful semantic difference. In order to improve classification, we decided to replace number expressions in tweets with a generic token.

#### Implementation Process
Numbers are replaced using a regular expression in the [RegexReplacer](https://github.com/team-one-ML/MLinPractice/blob/main/code/preprocessing/regex_replacer.py) class.

#### Discussion
Replacing numbers with a generic token has the advantage of removing unnecessary noise from the dataset in order to aid in classification, assuming that individual number expressions are irrelevant to the task. Since the dataset specifically encompasses tweets related to data science, there is a chance that tweeters will use numbers more frequently and that numbers have a higher significance to the tweet message, but we believe that the specific value of the number expression does not influence virality.

### URL removal
#### Goal
If a tweet in our dataset contains URLs, they have to be removed. This ensures that they do not influence feature extraction based on english language later on. For instance, we want to use features like named-entity-recognition and sentiment-analysis.
#### Implementation process
We used a regular expression in run_preprocessing.py to filter for URLs in the tweet and remove them. To do this, we implemented the class RegexReplacer in regex_replacer.py.
#### Discussion
**TODO**


### Lemmatization
#### Goal
Our goal was to generalize the form of words down to their lemmas. This enhances comparability between tweets and is an important precondition for stopword removal. 
#### Implementation process
We created the class "Lemmatizer" in lemmatizer.py which accesses the "WordNetLemmatizer" from nltk and used part-of-speech tags to replace the words in tweet by their lemmas.
#### Discussion
**TODO**


### Stopword removal
#### Goal
Our goal was to generalize the form of words down to their lemmas. This enhances comparability between tweets and is an important precondition for stopword removal. 
#### Implementation process
We created the class "Lemmatizer" in lemmatizer.py which accesses the "WordNetLemmatizer" from nltk and used part-of-speech tags to replace the words in tweet by their lemmas.
#### Discussion
**TODO**

## Feature Extraction

### DateTime Transformations
#### Goal
In the given dataset, the columns "Datetime" "Date" and "Time" contain information regarding the time the tweet was created. What they lack, are meaningful categorical subsets of time. The time of tweet creation may be a valuable feature for our model, but exact dates and seconds probably don't carry too much relevant information. Instead, we have transformed the data to include the month, weekday, season and time of day that the tweet was created. These features might hold more information as to the chance of a tweet becoming viral and are categorical, thus can be used by any classifier. 
#### Implementation Process
The Implementations can be found here: [month.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/month.py), [weekday.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/weekday.py), [season.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/season.py), [daytime.py](https://github.com/team-one-ML/MLinPractice/blob/main/code/feature_extraction/daytime.py). The respective classes retrieve the relevant column from the dataset, convert it into [Datetime](https://docs.python.org/3/library/datetime.html)] format and transform the times into categorical data.
#### Discussion
**TODO**: After choosing final features

### Sentiment Analysis

#### Goal
Our hypothesis is that emotionality and subjectivity of tweet content influences virality. We therefore choose to employ a sentiment analyzer to extract positive and negative sentiment from the tweets.

#### Implementation Process
We use the vader sentiment analyzer from the Natural Langauge Toolkit (NLTK) to extract postive, negative, neutral sentiment values that range from 0 to 1, as well as a compound value that ranges from -1 to 1. This analyzer is used in the [Sentiment feature extractor class](), which thus adds four output dimensions to the overall feature vector.

#### Discussion
Sentiment is often cited as one of the driving forces of content in social networks. We thus believe that it also plays a role in predicting tweet virality. The method of sentiment analysis used by the vader project, does not take into account sentence level semantics but merely word-level semantics. Specifically, it uses precalculated scores for the words it finds in the tweet to calculate an average sentiment for the whole text. This is a rather naive approach, but we believe it to be a worthwhile tradeoff between added value and performance.

### TF-IDF

#### Goal
In order to find words that are relevant for classification we use TF-IDF, which calculates the term frequency divided by the inverse document frequency. 

#### Implementation Process

####