#!/bin/bash

# execute the application with all necessary pickle files
# dim_red not used: data/dimensionality_reduction/pipeline.pickle
echo "Starting the application..."
python -m script.application.application data/preprocessing/pipeline.pickle data/feature_extraction/pipeline.pickle data/classification/classifier.pickle
