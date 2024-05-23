# Variation-Theory-in-Counterfactual-Data-Augmentation

## Quick Start
1. install the required packages using pip install -r requierments.txt
    make sure you have pytorch installed
2. replace <<API_KEY>> inside config.ini with your openAI API key

3. make sure the run_scripts.sh file has permission to execute by running chmod 

4. replace all required inputs surrounded by <<>> inside config.ini by following the comment instructions on each line
5. run the following command:
```./run_scripts.sh```


## Files
### 01_data_formatting
This file iteratively trains the symbolic model and generates patterns using the data's ground truth. The initial run may take several minutes due to caching, but subsequent runs should be quicker.

### 02_counterfactual_over_generation
This file uses the candidate phrases generated in the previous file to generate the counterfactual examples that will be used for finetuning a GPT-3.5.

### 03_counterfactual_filtering
This file uses the three level filters -- herustic, symbolic, and GPT-discriminator -- to assess the qualtity of previously generated counterfactuals.

### 04_fine_tuning
This file fine-tunes a GPT-3.5 model to generate counterfactual data.

### 05_AL_testing_BERT, 05_AL_testing
These files iteratevely train BERT and GPT models using the different cases and meaure the performance of the trained models.