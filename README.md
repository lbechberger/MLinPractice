# ML in Practice
Source code for the Seminar "Machine Learning in Practice", taught at Osnabrück University in the winter term 2021/2022.

As data source, we use the "Data Science Tweets 2010-2021" data set (version 3) by Ruchi Bhatia from [Kaggle](https://www.kaggle.com/ruchi798/data-science-tweets). The goal of our example project is to predict which tweets will go viral, i.e., receive many likes and retweets.

## Virtual Environment

In order to install all dependencies, please make sure that you have a local [Conda](https://docs.conda.io/en/latest/) distribution (e.g., Anaconda or miniconda) installed. Begin by creating a new environment called "MLinPractice" that has Python 3.6 installed:

```conda create -y -q --name MLinPractice python=3.6```

You can enter this environment with `conda activate MLinPractice` (or `source activate MLinPractice`, if the former does not work). You can leave it with `conda deactivate` (or `source deactivate`, if the former does not work). Enter the environment and execute the following commands in order to install the necessary dependencies (this may take a while):

```
conda install -y -q -c conda-forge scikit-learn=0.24.2
conda install -y -q -c conda-forge matplotlib=3.3.4
conda install -y -q -c conda-forge nltk=3.6.3
conda install -y -q -c conda-forge gensim=4.1.2
conda install -y -q -c conda-forge spyder=5.1.5
conda install -y -q -c conda-forge pandas=1.1.5
```

You can double-check that all of these packages have been installed by running `conda list` inside of your virtual environment. The Spyder IDE can be started by typing `~/miniconda/envs/MLinPractice/bin/spyder` in your terminal window (assuming you use miniconda, which is installed right in your home directory).

In order to save some space on your local machine, you can run `conda clean -y -q --all` afterwards to remove any temporary files.

The installed libraries are used for machine learning (`scikit-learn`), visualizations (`matplotlib`), NLP (`nltk`), word embeddings (`gensim`), and IDE (`spyder`), and data handling (`pandas`)

## Overall Pipeline

The script `code/load_data.sh` downloads the raw csv files containing the tweets and their metadata. They are stored in the folder `data/raw/` (which will be created if it does not yet exist).


**TODO:**
- Overall script
- preprocessing
- feature extraction
- dimensionality reduction
- training & hyperparameter optimization
- application

## Preprocessing

## Feature Extraction

## Dimensionality Reduction

## Classifier

## Application
