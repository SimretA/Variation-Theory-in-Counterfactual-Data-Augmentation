#!/bin/bash

set -x

# Extracting environment name from config.ini
environment=$(grep 'environment' config.ini | cut -d '=' -f2)
echo "Environment: $environment"

# remove the cache files if they exist
rm user_checkpoints/test.json

#run the files in the pipeline
conda run -n $environment python 01_data_formatting.py 2>&1

conda run -n $environment python 02_counterfactual_over_generation.py 2>&1

conda run -n $environment python 03_counterfactual_filtering.py 2>&1

conda run -n $environment python 04_fine_tuning.py 2>&1

conda run -n $environment python 05_AL_testing_BERT.py 2>&1

conda run -n $environment python 05_AL_testing.py 2>&1
