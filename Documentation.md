# Documentation

This is the forked repository for Magnus Müller, Maximilian Kalcher and Samuel Hagemann. 

Our task involved building and documenting a real-life application of a machine learning task. We were given a dataset of 295811 tweets about data science from the years 2010 until 2021 and had to build a classifier that would detect whether a tweet would go viral or not. The measure for it being viral was when the sum of likes and retweets were bigger than 50, which resulted in 91% false (or non-viral) labels, and 9% true (or viral) labels.

Our work consisted of a number of steps in a pipeline, in summary: We loaded and labeled the data using the framework given to us by Bechberger. We then preprocessed the data, mainly the raw tweet, to better fit our feature extraction later. This was done by removing punctuation, stopwords, etc. and also tokenizing into single words. After this, we extracted a handful of features which we found to be of importance, some were already included in the raw dataset columns, some we had to extract ourselves. Since the feature space was not exactly very large and mostly overseeable, we did not apply any dimensionality reduction other than what was already implemented. So after the features, we headed straight into classification using a variety of classifiers and benchmarks for evaluation. At the end, our best classifier is implemented into an 'application', callable by terminal, which gives the likeliness of an input tweet being viral, having used the features as training. 

This pipeline is documented more in detail below.

## Preprocessing

Before having used any data of columns such as the raw tweets, it was important for us to process parts of it beforehand so our chosen features could be extracted smoothly. The first thing we noticed was that many tweets had different kinds of punctuation, stopwords, emojis, and some of them even were written in different languages.

Since we found it interesting from the start to use some NLP features like the semantics of words alongside other features, we aimed to keep the core words of the tweet only.

### Design Decisions

To achieve to only keep the core words of a tweet, we used the following data cleaning methods: 

1. Removing punctuation and digits

We knew that tweets can sometimes use extensive punctuation, which would be a problem for later features and/or the tokenizer, since it detects punctuation as a token too. We chose to remove any written digits as well to only keep strings. With the help of the ```string``` package, we filtered out most punctuation and digits, namely: ```['!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789] ``` . But we also had to add some other special characters that the package did not pick up, and were still being present after the cleaning: ``` [’—”“↓'] ``` . 

Example: 

Input: 
` Red Black tree, AVL Tree and other Algorithms like Bellman ford etc are asked in most interviews even on Devops, Data Science roles 😏 httptcoQrKYJpiiVp `

Output: 
` Red Black tree AVL Tree and other Algorithms like Bellman ford etc are asked in most interviews even on Devops Data Science roles 😏 httptcoQrKYJpiiVp` 


2. Removing stopwords

For the sake of only using important words for our features, we removed so called stopwords, or filler words used commonly in sentences. This was possible with the ```nltk``` package. So for each word in the tweet, any word that was equal to the corpus of stopwords was removed :
```
{'a',
 'about',
 'above',
 'after',
 'again',
… 
 'your',
 'yours',
 'yourself',
 'yourselves'}
```
(179 total) 

Example: 

Input: 
`Red Black tree, AVL Tree and other Algorithms like Bellman ford etc are asked in most interviews even on Devops, Data Science roles 😏 httptcoQrKYJpiiVp`

Output: 
`Red Black tree, AVL Tree Algorithms like Bellman ford etc asked interviews even Devops, Data Science roles 😏 httptcoQrKYJpiiVp`


3. Remove emojis

In between the now usable words, we found some tweets had emojis. This of course was not a string and since interpretability of emojis by encoding and decoding into ASCII was a little difficult, we chose to just remove them from the tweet. 

Seeing as how the decoded emojis all start with ```\UXXXX``` followed by a set of numbers and letters, we just removed every string that started or contained ```\U``` after being decoded (contained - because some emojis were written without spacing and recognized as a single string). 

Example: 

Input: ``` Red Black tree, AVL Tree and other Algorithms like Bellman ford etc are asked in most interviews even on Devops, Data Science roles 😏 httptcoQrKYJpiiVp```

Output: ``` Red Black tree, AVL Tree and other Algorithms like Bellman ford etc are asked in most interviews even on Devops, Data Science roles httptcoQrKYJpiiVp```

4. Remove links

We thought we had finished the cleaning part, since there was finally only text in a tweet. But we quickly found that links were also recognized as strings, and were practically unusable for us. So we also removed any string that started with ``` http ```. 

Example: 

Input: 
`Red Black tree, AVL Tree and other Algorithms like Bellman ford etc are asked in most interviews even on Devops, Data Science roles 😏 httptcoQrKYJpiiVp`

Output: 
`Red Black tree, AVL Tree and other Algorithms like Bellman ford etc are asked in most interviews even on Devops, Data Science roles 😏`

5. Tokenize tweet

After these important cleaning steps, we took all words in the tweet and tokenized them for easier feature extraction when dealing with NLP features, because we would then just be iterating over a list or core words. This tokenizer is built using ```nltk``` and was already implemented in one of the seminar sessions. 

Every part of the string that is not a whitespace or ```’ ‘``` is then added onto a list of so-called tokens that represent the words in the sentence. We didn’t want to overwrite the normal preprocessed tweet, because of further features that would not need tokens, so we decided to just insert an extra preprocessing column only for these tokenized words. 

Example: 

Input: ``` Red Black tree, AVL Tree and other Algorithms like Bellman ford etc are asked in most interviews even on Devops, Data Science roles 😏 httptcoQrKYJpiiVp```

Output: ``` [‘Red’, ‘Black’, ‘tree’, ’,’, ‘AVL’, ‘Tree’, ‘and’, ‘other’, ‘Algorithms’, ‘like’, ‘Bellman’, ‘ford’, ‘etc’, ‘are’, ‘asked’, ‘in’, ‘most’, ‘interviews’, ‘even’, ‘on’, ‘Devops’, ‘,’, ‘Data’, ‘Science’, ‘roles’, ‘😏’, ‘httptcoQrKYJpiiVp’ ]```

6. Set language to only english

Taking a closer look at the languages of our tweets, our analysis summary was the following: 

{'en': 282035, 'it': 4116, 'es': 3272, 'fr': 2781, 'de': 714, 'id': 523, 'nl': 480, 'pt': 364, 'ca': 275, 'ru': 204, 'th': 157, 'ar': 126, 'tl': 108, 'tr': 84, 'hr': 68, 'da': 66, 'ro': 60, 'ja': 58, 'sv': 42, 'et': 29, 'pl': 25, 'bg': 24, 'af': 23, 'no': 21, 'fi': 20, 'so': 16, 'ta': 16, 'hi': 11, 'mk': 11, 'he': 9, 'sw': 9, 'lt': 7, 'uk': 6, 'sl': 6, 'te': 5, 'zh-cn': 5, 'lv': 5, 'ko': 5, 'bn': 4, 'el': 4, 'fa': 3, 'vi': 2, 'mr': 2, 'ml': 2, 'hu': 2, 'kn': 1, 'cs': 1, 'gu': 1, 'sk': 1, 'ur': 1, 'sq': 1}

It turns out, from the total 295811 samples, 95% were english tweets. The rest would be pretty much unusable for our subsequent NLP-based features, so we chose to remove that 5% portion from our preprocessed data. 

Example: 

Input: ``` Red Black tree, AVL Tree and other Algorithms like Bellman ford etc are asked in most interviews even on Devops, Data Science roles 😏 httptcoQrKYJpiiVp```

Output: ``` Red Black tree, AVL Tree and other Algorithms like Bellman ford etc are asked in most interviews even on Devops, Data Science roles 😏 httptcoQrKYJpiiVp```

7. Set all words to lowercase

Our final preprocessing step was only implemented after having examined keywords for the feature extraction and found out that many words on ‘most used words’ analysis had appeared twice. This was because sometimes people would write them with lower and uppercase. So we just went back into the code and made sure to append all strings to our preprocessing columns with ```.lower()```, making all words lowercase. 

Example: 

Input: ``` Red Black tree, AVL Tree and other Algorithms like Bellman ford etc are asked in most interviews even on Devops, Data Science roles 😏 httptcoQrKYJpiiVp```

Output: ``` red black tree, avl tree and other algorithms like bellman ford etc are asked in most interviews even on devops, data science roles 😏 httptcoQrKYJpiiVp```

### Results

At the end of our preprocessing, we had created 2 new columns: ``` preprocess_col ``` for the general feature extraction and ``` preprocess_col_tokenized ``` for the feature extraction based on the word level (semantics). 

Our resulting columns from a raw tweet looks like this: 

(raw) (preprocess_col) (preprocess_col_tokenized)
Looking at this from a different perspective, we can count how much of the original dataset was removed solely based on the character count: 

Number of characters before preprocessing: 52686072
Number of characters after preprocessing: 32090622
Removed: 20595450 (39.1 %)

At the end of our preprocessing, almost 60% of the original data was used and worked with. 

## Feature Extraction

The dataset was very variable and we had a lot of features to work with, which gave us the freedom to choose and experiment with these freely. We chose to extract two types of features: 

Features about special characters and metadata, and
Features about natural language, word similarity and frequency

These features are appended to the preprocessed dataset as new columns and can be inspected in the ``` features.csv ``` file in the feature extraction folder. Below we listed all extracted features and their explanation. 

### Design Decisions

Below are listed all implemented features and their explanation. 

Photo Bool

We implemented a simple feature that would tell whether the tweet under consideration contains one (or even multiple) photos or not. The idea behind this was that tweets with visual stimulus are more appealing and would therefore be more likely to go viral. 

The possible outputs are either ```[1]``` if the tweet contains photo(s) and ```[0]``` if it does not. If we have 5 tweets that either contain videos or not, an exemplary output could be: ```[[1], [0], [0], [1], [0]]```

Video Bool

Also, we were interested in the possible effect of video(s) in the tweets and implemented it as a feature in a similar way to the photo feature. This feature will tell whether the tweet under consideration contains video(s) or not. The possible outputs are either ```[1]``` if the tweet contains video(s) and ```[0]``` if it does not. If we have 5 tweets that either contain videos or not, an exemplary output could be: ```[[0], [1], [0], [1], [1]]```

Since this feature was already integrated in the raw dataset, we just had to extract the values of this column. 

Hashtag Counter

We also thought about the importance of hashtags. Since using a hashtag increases the engagement of tweets by tagging similar topics together, having more hashtags would mean that more people could be reached for a variety of topics. Because of this, we implemented a feature that counts the number of hashtags a tweet has. If we have 5 tweets that contain different number of hashtags, an exemplary output could be: ```[[0], [5], [2], [3], [1]]```

Emoji Counter

According to a study, emojis have not only become a visual language used to reveal several things (like feelings and thoughts), but also part of the fundamental structure of texts. They have shown to convey easier communication and strengthen the meaning and social connections between users. 

Because of this, we thought it would be very fitting to add a feature about emojis. So we took the original unprocessed tweets again, and counted every occurrence of strings that started or contained ```\U``` after being decoded. This number was then added to a new column as a feature to show that using a n > 0 number of emojis would increase the attractiveness of the tweet. If we have 5 tweets that contain different number of emojis, an exemplary output could be: ```[[0], [5], [2], [3], [1]]```


Tweet length

This feature was already implemented, so we did not add anything to it. We did find this very interesting though. According to a 2013 study trying to analyze why tweets became viral, most viral tweets back then had less than 140 characters. This means that a shorter and precise tweet would perform better than a longer tweet conveying more information.

This feature just counts the entire tweet string and adds the value to a new feature column. If we have 5 tweets of different length, an exemplary output could be: ```[[45], [14], [41], [30], [12]]``

Hour of tweet

The last feature about the tweet metadata is about the posting time of the tweet. We wanted to know if there was a difference in the posting times in viral and non-viral tweets. The following two graphs represent the tweet frequency per hour of posting. 

![Test](/Documentation/time_non_viral.png)
[2] Hour frequency of non-viral tweets [0-24]

[3] Hour frequency of viral tweets [0-24]

Interestingly enough, both viral and non-viral tweets are distributed pretty much equally except for a timeframe of about 3 hours in the morning from 7:00 - 10:00. This is where the viral tweets tend to be tweeted more. Using this information, we stripped the hour from the ```time``` column of the raw dataset and added this as a feature in a new column. So for a time of ```12:05:45``` the feature would just extract the number ```12```. If we have 5 tweets of posting times, an exemplary output could be: ```[[12], [14], [3], [4], [0]]``


Word2Vec

We then started to analyze features about semantics and natural language. The first thing that came to mind was seeing if there was a different word usage comparing the viral and non-viral tweets. The two graphs below show the 20 most used words in both categories:

[3] Top 20 most used words in non-viral tweets (label == False)

[3] Top 20 most used words in viral tweets (label == True)

What we can gather from this is that both categories seem to use the word ‘data’, followed by ‘datascience’  the most. This is not surprising, as the entire dataset is about data-science related tweets. Applying a word embedding feature using these words would not make much sense, as the feature would score equally high for the majority of both viral and non-viral tweets. So we had to differentiate between the categories and find words that were only present in the top 20 words in viral tweets, and not in non-virals. Since we used a dataset of embedded words from google news ( ```'word2vec-google-news-300'```), the keywords had to be present there and could not be too specific (deepleaning, datascience, etc. did not exist). We then settled with the following words: 

` keywords  = ['coding','free','algorithms','statistics'] `

These words were present in the dataset for word embeddings and were also exclusively used in viral tweets. 

Using the package ```gensim``` for computing word embeddings and semantic similarity, we could easily iterate over all words in ```preprocess_col_tokenized``` and compare each word there with each word in our ```keyword``` list. For every (tokenized) tweet, we took the mean of all similarity values and added this float number to a new feature column ```word2vec```. 


Hashing Vectorizer
Tf Idf

We wanted to try something we didn't hear in the lecture. Therefore, we used the HashingVectorizer from sklearn to create an individual hash for each tweet. For a sentence like 'I love Machine Learning', the output can look like [0.4, 0.3, 0.9, 0, 0.21], with length n representing the number of features. It's not very intuitive to humans why this works, but after a long time of version conflicts and other problems, we enjoyed the simplicity of using sklearn. 

Usage: `--hash_vec` 
and for number of features for hash vector edit HASH_VECTOR_N_FEATURES in util.py 

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

## Evaluation

### Design Decisions

Which evaluation metrics did you use and why? 
Which baselines did you use and why?

### Results

How do the baselines perform with respect to the evaluation metrics?

### Interpretation

Is there anything we can learn from these results?

## Tests

We have written tests for tfidf_vec and hash_vector, because even though the sklearn functions themselves naturally have many tests implemented, we want to double check that we are using them correctly and that we are getting the expected output. Therefore, especially 'test_result_shape' is very important, because it checks if the length of the output list matches the number of input elements.  

We added in run_classifier, a number of functions to run from the run_classifier_test.py which tests all classifiers, checks if the data is still equal length, if no classifier is written, try classifier fit, if not, give correct error output. 

## Project Organization
