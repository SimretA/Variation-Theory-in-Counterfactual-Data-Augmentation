#!/bin/bash

set -x

# Extracting environment name from config.ini
environment=$(grep 'environment' config.ini | cut -d '=' -f2)
echo "Environment: $environment"

# remove the cache files if they exist
rm user_checkpoints/test.json

#run the first file
conda run -n $environment python 01_data_formatting.py 2>&1

conda run -n $environment python 02_counterfactual_over_generation.py 2>&1