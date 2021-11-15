#!/bin/bash

# create directory if not yet existing
mkdir -p data/dimensionality_reduction/

# run dimensionality reduction on training set to fit the parameters
echo "  training set"
python -m src.dimensionality_reduction.reduce_dimensionality data/feature_extraction/training.pickle data/dimensionality_reduction/training.pickle -e data/dimensionality_reduction/pipeline.pickle --verbose

# run dimensionality reduction on validation set and test set (with pre-fit parameters)
echo "  validation set"
python -m src.dimensionality_reduction.reduce_dimensionality data/feature_extraction/validation.pickle data/dimensionality_reduction/validation.pickle -i data/dimensionality_reduction/pipeline.pickle
echo "  test set"
python -m src.dimensionality_reduction.reduce_dimensionality data/feature_extraction/test.pickle data/dimensionality_reduction/test.pickle -i data/dimensionality_reduction/pipeline.pickle
