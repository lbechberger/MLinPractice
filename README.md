# Machine Learning in Practice
Source code for the practical Seminar "Machine Learning in Practice", taught at Osnabrück University in the winter term 2021/2022 at the Insitute of Cognitive Science.

As data source, we use the "Data Science Tweets 2010-2021" data set (version 3) by Ruchi Bhatia from [Kaggle](https://www.kaggle.com/ruchi798/data-science-tweets). The goal of our example project is to predict which tweets will go viral, i.e., receive many likes and retweets.

## Virtual Environment

In order to install all necessary dependencies, please make sure that you have a local [Conda](https://docs.conda.io/en/latest/) distribution (e.g., Anaconda or miniconda) installed. Begin by creating a new environment called "MLinPractice" that has Python 3.6 installed:

```conda create -y -q --name MLinPractice python=3.6```

You can enter this environment with `conda activate MLinPractice` (or `source activate MLinPractice`, if the former does not work). You can leave it with `conda deactivate` (or `source deactivate`, if the former does not work). Enter the environment and execute the following commands in order to install the necessary dependencies (this may take a while):

```
conda install -y -q -c conda-forge scikit-learn=0.24.2
conda install -y -q -c conda-forge matplotlib=3.3.4
conda install -y -q -c conda-forge nltk=3.6.3
conda install -y -q -c conda-forge gensim=4.1.2
conda install -y -q -c conda-forge spyder=5.1.5
conda install -y -q -c conda-forge pandas=1.1.5
conda install -y -q -c conda-forge mlflow=1.20.2
conda install -y -q -c conda-forge vaderSentiment=3.3.2
```

You can double-check that all of these packages have been installed by running `conda list` inside of your virtual environment. The Spyder IDE can be started by typing `~/miniconda/envs/MLinPractice/bin/spyder` in your terminal window (assuming you use miniconda, which is installed right in your home directory).

In order to save some space on your local machine, you can run `conda clean -y -q --all` afterwards to remove any temporary files.

The installed libraries are used for machine learning (`scikit-learn`), visualizations (`matplotlib`), NLP (`nltk`), word embeddings (`gensim`), and IDE (`spyder`), and data handling (`pandas`)

## Setup

First if all, the shell script `src/setup.sh` needs to be run once before the actual `src/pipeline.sh` script or any other shell scripts can be executed. The setup script downloads necessary data by executing the scripts `src/load_data.sh` and `src/load_nltk_data.sh`.  
- The former script `src/load_data.sh` downloads the Data Science Tweets as raw csv files containing the tweets and their metadata. They are stored in the directory `data/raw/` (which will be created if it does not yet exist).
- The latter script `src/load_nltk_data.sh` downloads necessary NLTK data sets, corpora and models (see more: [nltk.org/data.html](https://www.nltk.org/data.html))

## Running Scripts and Unit Tests

To run bash scripts you need to open a bash shell. On Unix systems (Linux and MacOS) such a shell comes already with the operating system. On Windows this needs to be installed manually. When you install git, the git bash shell will be installed as well. Once you open a terminal window, you can either directly write the path to the script you want to execute or preprend `bash` before it. Both of the following example commands should work:

```console
./src/setup.sh
```

```console
bash ./src/setup.sh
```

In case this throws an error like `permission denied` or sth similar, you might need to change the access level of some files. This can be done by executing the following command:

```console
chmod -R a+x ./src
```

This gives access rights to all users for all files (recursively) in the src directory.

### Pipeline Scripts

The overall pipeline can be executed with the script `src/pipeline.sh`, which executes all of the following shell scripts:
- `src/preprocessing.sh`: Executes all necessary preprocessing steps, including a creation of labels and splitting the data set.
- `src/feature_extraction.sh`: Takes care of feature extraction.
- `src/dimensionality_reduction.sh`: Takes care of dimensionality reduction.
- `src/classification.sh`: Takes care of training and evaluating a classifier. The scripts specifies one of 5 possible classification scenarios. 4 of them are commented out. Comment in or out the so that only one scenario will be run for training on the training set. Additionally the same classifier is used for evaluation on the validation set.

### Additional Scripts
- `src/application.sh`: Launches the application example.
- `src/classification_hyper_param.sh`: Trains and evaluates two classifiers over a predefined range of parameters (grid search)
- `src/final_classification.sh`: Trains the best two classifiers on the training .data set and afterwards evaluates the performance on the test data set in comparison to the *stratified* baseline.
- `src/setup.sh`: As mentioned above in detail, downloads necessary data.

### Unit Tests 

The following command runs all unit tests in the `src`directory for files that end in the file name `_test.py`:

```bash
python -m unittest discover -s src -p '*_test.py'
```

## Preprocessing

All python scripts and classes for the preprocessing of the input data can be found in [`src/preprocessing/`](src/preprocessing/).

### Creating Labels

The script [`create_labels.py`](src/preprocessing/create_labels.py) assigns labels to the raw data points based on a threshold on a linear combination of the number of likes and retweets. It is executed as follows:

```bash
python -m src.preprocessing.create_labels path/to/input_dir path/to/output.csv
```

Here, `input_dir` is the directory containing the original raw csv files, while `output.csv` is the single csv file where the output will be stored.

The script takes the following optional parameters:
- `-l` or `--likes_weight` determines the relative weight of the number of likes a tweet has received. Defaults to 1.
- `-r` or `--retweet_weight` determines the relative weight of the number of retweets a tweet has received. Defaults to 1.
- `-t` or `--threshold` determines the threshold a data point needs to surpass in order to count as a "viral" tweet. Defaults to 50.

### Classical Preprocessing

The script [`run_preprocessing.py`](src/preprocessing/run_preprocessing.py) is used to run various preprocessing steps on the raw data, producing additional columns in the csv file. It is executed as follows:

```bash
python -m src.preprocessing.run_preprocessing path/to/input.csv path/to/output.csv
```

Here, `input.csv` is a csv file (ideally the output of `create_labels.py`), while `output.csv` is the csv file where the output will be written.

The following flags configure which preprocessing steps are applied:

- `-p` or `--punctuation`: A new column *"tweet_no_punctuation"* is created, where all punctuation is removed from the original tweet. (See [punctuation_remover.py](src/preprocessing/preprocessors/punctuation_remover.py) for more details)
- `-t` or `--tokenize`: Tokenize the given column (can be specified by `--tokenize_input`, default = "tweet"), and create new column with suffix "_tokenized" containing tokenized tweet.
- `-o` or `--other`: Executes all the other preprocessing steps like the removal of non english records and the removal of unnecessary columns.

Moreover, the script accepts the following optional parameters:

- `-e` or `--export` gives the path to a pickle file where an sklearn pipeline of the different preprocessing steps will be stored for later usage.

### Splitting the Data Set

The script [`split_data.py`](src/preprocessing/split_data.py) splits the overall preprocessed data into training, validation, and test set. It can be invoked as follows:

```bash
python -m src.preprocessing.split_data path/to/input.csv path/to/output_dir
```

Here, `input.csv` is the input csv file to split (containing a column "label" with the label information, i.e., `create_labels.py` needs to be run beforehand) and `output_dir` is the directory where three individual csv files `training.csv`, `validation.csv`, and `test.csv` will be stored.
The script takes the following optional parameters:
- `-t` or `--test_size` determines the relative size of the test set and defaults to 0.2 (i.e., 20 % of the data).
- `-v` or `--validation_size` determines the relative size of the validation set and defaults to 0.2 (i.e., 20 % of the data).
- `-s` or `--seed` determines the seed for intializing the random number generator used for creating the randomized split. Using the same seed across multiple runs ensures that the same split is generated. If no seed is set, the current system time will be used.


## Feature Extraction

All python scripts and classes for feature extraction can be found in [`src/feature_extraction/`](src/feature_extraction).

The script [`extract_features.py`](src/feature_extraction/extract_features.py) takes care of the overall feature extraction process and can be invoked as follows:

```bash
python -m src.feature_extraction.extract_features path/to/input.csv path/to/output.pickle
```

Here, `input.csv` is the respective training, validation, or test set file created by `split_data.py`. The file `output.pickle` will be used to store the results of the feature extraction process, namely a dictionary with the following entries:
- `"features"`: a numpy array with the raw feature values (rows are training examples, colums are features)
- `"feature_names"`: a list of feature names for the columns of the numpy array
- `"labels"`: a numpy array containing the target labels for the feature vectors (rows are training examples, only column is the label)

The features to be extracted can be configured with the following optional parameters:
- `-c` or `--char_length`: Count the number of characters in the "tweet" column of the data frame. (see [`character_length.py`](src/feature_extraction/feature_extractors/character_length.py))

Moreover, the script support importing and exporting fitted feature extractors with the following optional arguments:
- `-i` or `--import_file`: Load a configured and fitted feature extraction from the given pickle file. Ignore all parameters that configure the features to extract.
- `-e` or `--export_file`: Export the configured and fitted feature extraction into the given pickle file.

## Dimensionality Reduction

All python scripts and classes for dimensionality reduction can be found in [`src/dimensionality_reduction/`](src/dimensionality_reduction/).

The script [`reduce_dimensionality.py`](src/dimensionality_reduction/reduce_dimensionality.py) takes care of the overall dimensionality reduction procedure and can be invoked as follows:

```
python -m src.dimensionality_reduction.reduce_dimensionality path/to/input.pickle path/to/output.pickle
```

Here, `input.pickle` is the respective training, validation, or test set file created by `extract_features.py`. 
The file `output.pickle` will be used to store the results of the dimensionality reduction process, containing `"features"` (which are the selected/projected ones) and `"labels"` (same as in the input file).

The dimensionality reduction method to be applied can be configured with the following optional parameters:
- `-m` or `--mutual_information`: Select the `k` best features (where `k` is given as argument) with the Mutual Information criterion

Moreover, the script support importing and exporting fitted dimensionality reduction techniques with the following optional arguments:
- `-i` or `--import_file`: Load a configured and fitted dimensionality reduction technique from the given pickle file. Ignore all parameters that configure the dimensionality reduction technique.
- `-e` or `--export_file`: Export the configured and fitted dimensionality reduction technique into the given pickle file.

Finally, if the flag `--verbose` is set, the script outputs some additional information about the dimensionality reduction process.

## Classification

All python scripts and classes for classification can be found in [`src/classification/`](src/classification/).

### Train and Evaluate a Single Classifier

The script [`run_classifier.py`](src/classification/run_classifier.py) can be used to train and/or evaluate a given classifier. It can be executed as follows:

```
python -m src.classification.run_classifier path/to/input.pickle
```

Here, `input.pickle` is a pickle file of the respective data subset, produced by either `extract_features.py` or `reduce_dimensionality.py`. 

Support **importing and exporting trained classifiers** with the following optional arguments:
- `-i` or `--import_file`: Load a trained classifier from the given pickle file. Ignore all parameters that configure the classifier to use and don't retrain the classifier.
- `-e` or `--export_file`: Export the trained classifier into the given pickle file.


By default, this data is used to train a **classifier**. It is possible to chose 1 of 5 different scenarios for training. Either select one of two Two dummy classifiers as a baseline or the knn or random forest classifier. For the random forest classifier it can be additionally specified to either perform a grid search or not.

Dummy Classifier (baselines)
- `-d` or `--dummyclassifier` followed by either `most_frequent` or `stratified`
  - `most_frequent` is a [_DummyClassifier_](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) which always predicts the most frequently occuring label in the training set.
  - `stratified` is a [_DummyClassifier_](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) that makes predictions based on the label frequency in the training data (respects the training set’s class distribution).

Chose one of three possible options for the classification:
- `--knn` followed by the a interger number of k
- `-r` or `--randomforest` followed by a integer number of trees
  - if `--sk_gridsearch_rf` is omitted a normal random forest classifier with the provided number of trees will be used for training.
  - if `--sk_gridsearch_rf` is present, a grid search on a random forest classifier with a predefined (hardcoded) range of parameters is performed. Als the the number of trees is still expected, but will be ignored.

**Evaluation metrics** are then used by the classifier. Which metrics to use for evaluation can be specified with the following optional arguments:
- `-m` or `--metrics` followed by another option (default is `kappa`):
  - `none` no metrics will be used
  - `all` all metrics will be used
  - `accuracy`: Classification accurracy (i.e., percentage of correctly classified examples).
  - `kappa`: Cohen's kappa (i.e., adjusting accuracy for probability of random agreement).
  - `precision`: Precision (ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative)
  - `recall` Recall (the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples)
  - `f1`: F1-score (weighted average of precision and recall)
  - `jaccard`: Jaccard score (the size of the intersection divided by the size of the union of two label sets)

For more details on the metrics used, see: https://scikit-learn.org/stable/modules/classes.html#classification-metrics

Logging with MlFlow:

- `--log_folder` specifies where MlFlow will store its logging files. Default is `data/classification/mlflow`.
- `-n` or `--run_name` specifies a name for the classification run, so that runs can be identified afterwards when looking at the results in the MlFlow user interface.

Finally, the optional argument `-s` or `--seed` determines the seed for intializing the random number generator (which may be important for some classifiers). 
Using the same seed across multiple runs ensures reproducibility of the results. If no seed is set, the current system time will be used.

## Application

All python code for the application demo can be found in [`src/application/`](src/application/).

The script [`application.py`](src/application/application.py) provides a simple command line interface, where the user is asked to type in their prospective tweet, which is then analyzed using the trained ML pipeline.
The script can be invoked as follows:

```
python -m src.application.application path/to/preprocessing.pickle path/to/feature_extraction.pickle path/to/dimensionality_reduction.pickle path/to/classifier.pickle
```

The four pickle files correspond to the exported versions for the different pipeline steps as created by `run_preprocessing.py`, `extract_features.py`, `reduce_dimensionality.py`, and `run_classifier.py`, respectively, with the `-e` option.


## Running MlFlow

To look at the MlFlow results run the following command. This will host a local server on [http://127.0.0.1:5000](http://127.0.0.1:5000). Opening it displays the results of all previous runs on a web page. The runs can also be exported as csv files.

```
mlflow ui --backend-store-uri data/classification/mlflow
```

Mlflow allows us to specify an SQL like search for specific data.
For example the `params.classifier = "knn"` to search for all entries where a knn classifier was used.

Here is another examples to only display runs with a randomforest classifier on a validation set:

```
params.classifier = "randomforest" AND params.dataset = "validation"
```

More information on: [mlflow.org/docs/latest/search-syntax.html#syntax](https://www.mlflow.org/docs/latest/search-syntax.html#syntax)

## Debugging in Visual Studio Code

1. Running a file in debug mode configured as waiting, because otherwise it woulk just finish to quickly

```
python -m debugpy --wait-for-client --listen 5678 .\src\feature_extraction\test\feature_extraction_test.py
```

2. `launch.json` configuration to attach the editor to the already started debug process.

```json
"configurations": [
  {            
      "name": "Python: Attach",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      }            
  },
]
```

3. Start the attach debug configuration via the VS Code UI ([F5] key or `Run`/`Run and Debug` menu)
