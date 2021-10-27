# Machine Learning in Practice

Source code for the practical Seminar "Machine Learning in Practice", taught at Osnabrück University in the winter term 2021/2022 at the Insitute of Cognitive Science.

As data source, we use the "Data Science Tweets 2010-2021" data set (version 3) by Ruchi Bhatia from [Kaggle](https://www.kaggle.com/ruchi798/data-science-tweets). The goal of our example project is to predict which tweets will go viral, i.e., receive many likes and retweets.

## Virtual Environment

In order to install all necessary dependencies, please make sure that you have a local [Conda](https://docs.conda.io/en/latest/) distribution (e.g., Anaconda or miniconda) installed. Begin by creating a new environment called "MLinPractice" that has Python 3.6 installed:

```
conda create -y -q --name MLinPractice python=3.6
```

You can enter this environment with `conda activate MLinPractice` (or `source activate MLinPractice` , if the former does not work). You can leave it with `conda deactivate` (or `source deactivate` , if the former does not work). Enter the environment and execute the following commands in order to install the necessary dependencies (this may take a while):

```

conda install -y -q -c conda-forge scikit-learn=0.24.2
conda install -y -q -c conda-forge matplotlib=3.3.4
conda install -y -q -c conda-forge nltk=3.6.3
conda install -y -q -c conda-forge gensim=4.1.2
conda install -y -q -c conda-forge spyder=5.1.5
conda install -y -q -c conda-forge pandas=1.1.5
conda install -y -q -c conda-forge mlflow=1.20.2
conda install -y -q -c conda-forge spacy

```

You can double-check that all of these packages have been installed by running `conda list` inside of your virtual environment. The Spyder IDE can be started by typing `~/miniconda/envs/MLinPractice/bin/spyder` in your terminal window (assuming you use miniconda, which is installed right in your home directory).

In order to save some space on your local machine, you can run `conda clean -y -q --all` afterwards to remove any temporary files.

The installed libraries are used for machine learning ( `scikit-learn` ), visualizations ( `matplotlib` ), NLP ( `nltk` ), word embeddings ( `gensim` ), and IDE ( `spyder` ), and data handling ( `pandas` )

## Overall Pipeline

The overall pipeline can be executed with the script `code/pipeline.sh` , which executes all of the following shell scripts:
* The script `code/load_data.sh` downloads the raw csv files containing the tweets and their metadata. They are stored in the folder `data/raw/` (which will be created if it does not yet exist).
* The script `code/preprocessing.sh` executes all necessary preprocessing steps, including a creation of labels and splitting the data set.
* The script `code/feature_extraction.sh` takes care of feature extraction.
* The script `code/dimensionality_reduction.sh` takes care of dimensionality reduction.
* The script `code/classification.sh` takes care of training and evaluating a classifier.
* The script `code/application.sh` launches the application example.

* The script `code/all_in_one_multiple_input_features.sh`is a powerfull extra file, which will launch a entire sklearn pipeline.

## Preprocessing

All python scripts and classes for the preprocessing of the input data can be found in `code/preprocessing/` .

### Creating Labels

The script `create_labels.py` assigns labels to the raw data points based on a threshold on a linear combination of the number of likes and retweets. It is executed as follows:
```python -m code.preprocessing.create_labels path/to/input_dir path/to/output.csv

```

Here, `input_dir` is the directory containing the original raw csv files, while `output.csv` is the single csv file where the output will be written.
The script takes the following optional parameters:
* `-l` or `--likes_weight` determines the relative weight of the number of likes a tweet has received. Defaults to 1.
* `-r` or `--retweet_weight` determines the relative weight of the number of retweets a tweet has received. Defaults to 1.
* `-t` or `--threshold` determines the threshold a data point needs to surpass in order to count as a "viral" tweet. Defaults to 50.

### Classical Preprocessing

The script `run_preprocessing.py` is used to run various preprocessing steps on the raw data, producing additional columns in the csv file. It is executed as follows:
```

python -m code.preprocessing.run_preprocessing path/to/input.csv path/to/output.csv

```
Here, `input.csv` is a csv file (ideally the output of `create_labels.py`), while `output.csv` is the csv file where the output will be written.
The preprocessing steps to take can be configured with the following flags:
- `-p` or `--punctuation`: A new column "tweet_no_punctuation" is created, where all punctuation is removed from the original tweet. (See `code/preprocessing/punctuation_remover.py` for more details)
- `-t`or `--tokenize`: Tokenize the given column (can be specified by `--tokenize_input`, default = "tweet"), and create new column with suffix "_tokenized" containing tokenized tweet.
- `-s` or `--strings`: Remove stopwords, links and emojis in the "tweet" column of the data frame. (see code/preprocessing/string_remover.py)
- `--language`: Only keep tweets with the given language argument. Supported strings have to match enires of the "language" column of the data frame (en, fr, de, etc.). This argument is set to keep only english tweets (en). 

Moreover, the script accepts the following optional parameters:
- `-e` or `--export` gives the path to a pickle file where an sklearn pipeline of the different preprocessing steps will be stored for later usage.

### Splitting the Data Set

The script `split_data.py` splits the overall preprocessed data into training, validation, and test set. It can be invoked as follows:
```python -m code.preprocessing.split_data path/to/input.csv path/to/output_dir```

Here, `input.csv` is the input csv file to split (containing a column "label" with the label information, i.e., `create_labels.py` needs to be run beforehand) and `output_dir` is the directory where three individual csv files `training.csv` , `validation.csv` , and `test.csv` will be stored.
The script takes the following optional parameters:
* `-t` or `--test_size` determines the relative size of the test set and defaults to 0.2 (i.e., 20 % of the data).
* `-v` or `--validation_size` determines the relative size of the validation set and defaults to 0.2 (i.e., 20 % of the data).
* `-s` or `--seed` determines the seed for intializing the random number generator used for creating the randomized split. Using the same seed across multiple runs ensures that the same split is generated. If no seed is set, the current system time will be used.

## Feature Extraction

All python scripts and classes for feature extraction can be found in `code/feature_extraction/` .

The script `extract_features.py` takes care of the overall feature extraction process and can be invoked as follows:

```
python -m code.feature_extraction.extract_features path/to/input.csv path/to/output.pickle
```

Here, `input.csv` is the respective training, validation, or test set file created by `split_data.py` . The file `output.pickle` will be used to store the results of the feature extraction process, namely a dictionary with the following entries:
* `"features"`: a numpy array with the raw feature values (rows are training examples, colums are features)
* `"feature_names"`: a list of feature names for the columns of the numpy array
* `"labels"`: a numpy array containing the target labels for the feature vectors (rows are training examples, only column is the label)

The features to be extracted can be configured with the following optional parameters:
* `-c` or `--char_length`: Count the number of characters in the "tweet" column of the data frame. (see code/feature_extraction/character_length.py)
* `--photo_bool`: Tells whether the tweet contains photos or not from column "photos". (see code/feature_extraction/photo_bool.py)
* `--video_bool`: Tells whether the tweet contains videos or not from column "videos". (see code/feature_extraction/video_bool.py)
* `--time`: Extract what hour the tweet was sent from column "time". (see code/feature_extraction/time_feature.py)
* `--emoji_count`: Count the emojis in a tweet from column "tweet". (see code/feature_extraction/emoji_count.py)
* `--hashtags`: Count the hashtags in a tweet from column "hashtags". (see code/feature_extraction/hashtag_count.py)
* `--word2vec`: Compute the semantic distance of words to given keywords from column "preprocess_col_tokenized". (see code/feature_extraction/word2vec.py)
* `--hash_vec`: Compute the HashingVectorizer from column "preprocess_col" and set number of features in argument HASH_VECTOR_N_FEATURES in util.py. (see code/feature_extraction/hash_vector.py)
* `--tfidf_vec`: Compute the Tf Idf vector of the tweet "preprocess_col". (see code/feature_extraction/tfidf_vecotr.py)

Moreover, the script support importing and exporting fitted feature extractors with the following optional arguments:
* `-i` or `--import_file`: Load a configured and fitted feature extraction from the given pickle file. Ignore all parameters that configure the features to extract.
* `-e` or `--export_file`: Export the configured and fitted feature extraction into the given pickle file.

## Dimensionality Reduction

All python scripts and classes for dimensionality reduction can be found in `code/dimensionality_reduction/` .

The script `reduce_dimensionality.py` takes care of the overall dimensionality reduction procedure and can be invoked as follows:

```python -m code.dimensionality_reduction.reduce_dimensionality path/to/input.pickle path/to/output.pickle

```

Here, `input.pickle` is the respective training, validation, or test set file created by `extract_features.py` . 
The file `output.pickle` will be used to store the results of the dimensionality reduction process, containing `"features"` (which are the selected/projected ones) and `"labels"` (same as in the input file).

The dimensionality reduction method to be applied can be configured with the following optional parameters:
* `-m` or `--mutual_information`: Select the `k` best features (where `k` is given as argument) with the Mutual Information criterion

Moreover, the script support importing and exporting fitted dimensionality reduction techniques with the following optional arguments:
* `-i` or `--import_file`: Load a configured and fitted dimensionality reduction technique from the given pickle file. Ignore all parameters that configure the dimensionality reduction technique.
* `-e` or `--export_file`: Export the configured and fitted dimensionality reduction technique into the given pickle file.

Finally, if the flag `--verbose` is set, the script outputs some additional information about the dimensionality reduction process.

## Classification

All python scripts and classes for classification can be found in `code/classification/` .

### Train and Evaluate a Single Classifier

The script `run_classifier.py` can be used to train and/or evaluate a given classifier. It can be executed as follows:
```python -m code.classification.run_classifier path/to/input.pickle

```

Here, `input.pickle` is a pickle file of the respective data subset, produced by either `extract_features.py` or `reduce_dimensionality.py` . 

By default, this data is used to train a classifier, which is specified by one of the following optional arguments:
* `-m` or `--majority`: Majority vote classifier that always predicts the majority class.
* `-f` or `--frequency`: Dummy classifier that makes predictions based on the label frequency in the training data.

The classifier is then evaluated, using the evaluation metrics as specified through the following optional arguments:
* `-a`or `--accuracy`: Classification accurracy (i.e., percentage of correctly classified examples).
* `-k`or `--kappa`: Cohen's kappa (i.e., adjusting accuracy for probability of random agreement).
* `--small 1000`: use just 1000 tweets.

Moreover, the script support importing and exporting trained classifiers with the following optional arguments:
* `-i` or `--import_file`: Load a trained classifier from the given pickle file. Ignore all parameters that configure the classifier to use and don't retrain the classifier.
* `-e` or `--export_file`: Export the trained classifier into the given pickle file.

Finally, the optional argument `-s` or `--seed` determines the seed for intializing the random number generator (which may be important for some classifiers). 
Using the same seed across multiple runs ensures reproducibility of the results. If no seed is set, the current system time will be used.

## Application

All python code for the application demo can be found in `code/application/` .

The script `application.py` provides a simple command line interface, where the user is asked to type in their prospective tweet, which is then analyzed using the trained ML pipeline.
The script can be invoked as follows:
`` `python -m code.application.application path/to/preprocessing.pickle path/to/feature_extraction.pickle path/to/dimensionality_reduction.pickle path/to/classifier.pickle` ``

The four pickle files correspond to the exported versions for the different pipeline steps as created by `run_preprocessing.py` , `extract_features.py` , `reduce_dimensionality.py` , and `run_classifier.py` , respectively, with the `-e` option.

## Complete Sklearn Pipeline

When you first see this project, it might be a little confusing how everything works together and what output you should put as input in your next file. If you feel that way, you can look at the file `code/all_in_one_multiple_input_features.py` , which you can run with `code/all_in_one_multiple_input_features.sh` . In this file you can do feature extraction, dimensionality reduction and classification at the same time.
You have many options to choose from.
Execute it with the script `code/all_in_one_multiple_input_features.sh` :
* In `classifieres=("SGDClassifier LogisticRegression LinearSVC")` you can specify all the classifiers you want to try.
* `python3 -m code.all_in_one_multiple_input_features data/preprocessing/preprocessed.csv -e data/classification/classifier.pickle --accuracy --kappa --balanced_accuracy --classification_report --classifier $k --feature_extraction 'union' --verbose`
* The script needs a csv input file generated e.g. by your preprocessing script. It is also possible to enter your raw labeled data.
* `-e` or `--export_file` for your output file of the classifier.
* You can select some or several metrics:
    - `--accuracy`
    - `--kappa`
    - `--balanced_accuracy`: A balanced_accuracy of 0.5 means that your classifier is just guessing. This is sometimes more helpful than your base accuracy, because the base accuracy can be 0.9, but your classifier still hasn't learned anything.
    - `--classification_report` creates a table with precision, recall and f1 score for all data and separately for each class.
* `--small 1000` if you don`t want to use all data, but only some (e.g. 1000) for testing.
* `--verbose` if you use this, get some information during training.
* feature_extraction:
    - `--feature_extraction` (expects a string, e.g. `--feature_extraction` `union` ) choose between `HashingVectorizer`, `TfidfVectorizer` and `union`. With union you can combine many features, like: 
        - are there videos / photos
        - when was the post (time feature)
        - length of the tweet
        - tweet text combined with e.g. HashingVectorizer  
    - Check out the FeatureUnion part of the script for even more custom selections.
* `--dim_red` for custom dimensionality reduction
    - `--dim_red` `SelectKBest(mutual_info_regression)`
    - `--dim_red` `SelectKBest(chi2)`
    but this is not recommended here, especially when using something like HashingVectorizer with 2 ** 22 features, because of time, too much information loss and because during our tests the classifier could easily learn this without dim_red.

* `--classifier`expects a string:
    - `--classifier` `SGDClassifier`
    - `--classifier` `MultinomialNB`
    - `--classifier` `LogisticRegression`
    - `--classifier` `LinearSVC`
    - `--classifier` `SVC`
