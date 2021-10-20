#!/bin/bash

# create directory if not yet existing
mkdir -p data/feature_extraction/

# run feature extraction on training set (may need to fit extractors)

echo -e "\n -> training set"
python -m code.feature_extraction.extract_features data/preprocessing/split/training.csv data/feature_extraction/training.pickle -e data/feature_extraction/pipeline.pickle --char_length --photo_bool --photo_bool --video_bool --replies_count --time --word2vec --emoji_count

# run feature extraction on validation set and test set (with pre-fit extractors)
echo -e "\n -> validation set"
python -m code.feature_extraction.extract_features data/preprocessing/split/validation.csv data/feature_extraction/validation.pickle -i data/feature_extraction/pipeline.pickle --char_length --photo_bool --photo_bool --video_bool --replies_count --time --word2vec --emoji_count

echo -e "\n -> test set\n"
python -m code.feature_extraction.extract_features data/preprocessing/split/test.csv data/feature_extraction/test.pickle -i data/feature_extraction/pipeline.pickle --char_length --photo_bool --photo_bool --video_bool --replies_count --time --word2vec --emoji_count

