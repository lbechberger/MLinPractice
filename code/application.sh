#!/bin/bash

# execute the application with all necessary pickle files
echo "Starting the application..."
python -m code.application.application data/preprocessing/pipeline.pickle data/feature_extraction/pipeline.pickle data/dimensionality_reduction/pipeline.pickle data/classification/classifier.pickle