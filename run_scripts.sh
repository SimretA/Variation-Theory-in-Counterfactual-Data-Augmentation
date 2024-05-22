#!/bin/bash

# Extracting environment name from config.ini
environment=$(grep 'environment' config.ini | cut -d '=' -f2)
echo "Environment: $environment"

# remove the cache files if they exist
rm user_checkpoints/test.json

#run the first file
data_file=$(grep 'data_file' config.ini | cut -d '=' -f2)
conda run -n $environment python 01_data_formatting.py $data_file