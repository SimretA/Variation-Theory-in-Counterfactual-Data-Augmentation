import logging
import os

import random
import pandas as pd
import sys
import copy
from sklearn.metrics import f1_score

import time

from openai import OpenAI
import json


#fine-tuned modesl - Yelp => ft:gpt-3.5-turbo-0613:university-of-notre-dame::8US4bHqN
#                   - massive => ft:gpt-3.5-turbo-0613:university-of-notre-dame::8UH39TNK


seed = 1

client = OpenAI(
    api_key="<<API_KEY>>"
)

#use the filtered dataset to finetune a smaller GPT model


#use finetuned model to generate counterfactauls
def generate_counterfactual_withmodel(modelname = None):

    test_prompt = 'good service and delicious food !\nOriginal label: service\nPattern: ADJ+*+(service) \nHighlight: "good servide" \n Candidate phrases: [great vibes, awful smell, good atmosphere] \nTarget label: environment\n\n###\n\n'

    response = client.chat.completions.create(
        model=modelname,
        messages=[{"role":"user", "content":test_prompt}]
        )
    
    return response.choices[0].message.content
    


def fine_tune_model(file_path, candidate_phrases_path):

    #load the filtered dataset
    data = pd.read_csv(file_path)
    
    #load the candidate phrases
    candidate_phrases = pd.read_csv(candidate_phrases_path)

    prompts = []
    messages = []
    completions = []

    

    for i, row in data.iterrows():

        # check if the the row has gone through the three filters
        passed_filter = row['matched_pattern'] and row['heuristic_filtered'] and row['is_target'] and (not row['is_ori'])

        # id,ori_text,ori_label,pattern,highlight,candidate_phrases,target_label,counterfactual,heuristic_filtered,matched_pattern,is_ori,is_target

        
        if not passed_filter:
            continue
        


        message = []

        message.append({"role":"system" ,"content":"The assistant will generate a counterfactual example close to the original sentence that contains one of the given candidate phrases."})
        message.append({"role":"system" ,
                        "content":f'''
                        The system will change the given sentence from the current label to the target.
                        For example: 'Find me a train ticket next monday to new york city' with original label:transport would be turned to 'Play me a song called New York City by Taylor Swift' with a label audio.
                        The system must use one of the provided candidate phrases to help you generate the counterfactuals.
                        Please make the sentnece about the target_label. Make sure that the new sentence is not about the original label.
                        You must use one of the candidate phrases without rewording it in the new sentence based on the following three criteria:
                        criteria 1: the phrase should change the label from the original label to the target_label to the highest degree. 
                        criteria 2: the modified sentence can not also be about the original label.
                        criteria 3: the modified sentence should be grammatically correct.
                    '''})
        id = row['id']
        #select a row with the same id from candidate_phrases
        target_label = row['target_label']
        pattern = row['pattern']

        highlight = row['highlight']
        candidate_phrase = row['candidate_phrases']
        try:
            text = data['ori_text'][i].strip() + '\nOriginal label: ' + row['ori_label'] + '\nPattern: ' + pattern +  '\nHighlight: ' + highlight + '\nCandidate phrases: ' + candidate_phrase  + '\nTarget label: ' + target_label+ '\n\n###\n\n'

        except:
            continue
        outcome = " "+ row['counterfactual'].strip() + "###"
        message.append({"role":"user" ,"content":f"{text}"})
        message.append({"role":"assistant" ,"content":f"{outcome}"})
        messages.append({"messages":message})
        # prompts.append(text)
        # completions.append(outcome)
        
    #df is the data we will use to finetune a gpt model    
    df = pd.DataFrame(zip(prompts,completions),columns=['prompt','completion'])
    
    finetuningdata_path = "output_data/YELP_GPT35_fine_tune_data.jsonl"
   
    df.to_json( finetuningdata_path,orient='records',lines=True)

    with open(finetuningdata_path, 'w') as f:
        for message in messages:
            # Convert each dictionary to a JSON string and write it to the file with a newline
            json.dump(message, f)
            f.write('\n')
    
    

    uploaded_data = client.files.create(
        file=open(finetuningdata_path, "rb"),
        purpose="fine-tune"
    )

    print(f"Data uploaded with ID: {uploaded_data.id}")


    #this is where we fine-tune the model
    model_engine = "gpt-3.5-turbo" #"davinci-002"
    n_epochs = 3
    batch_size = 4
    learning_rate = 1e-5
    max_tokens = 1024

    # Create the fine-tuning job
    fine_tuning_job = client.fine_tuning.jobs.create(
        model=model_engine,
        training_file=uploaded_data.id,
    )

    job_id = fine_tuning_job.id
    print(f"Fine-tuning job created with ID: {job_id}")

    # Wait for the fine-tuning job to complete
    while True:
        fine_tuning_status = client.fine_tuning.jobs.retrieve(job_id)
        print(fine_tuning_status)
        status = fine_tuning_status.status
        print(f"Fine-tuning job status: {status}")

        if status in ["completed", "failed"]:
            break

        time.sleep(120)



if __name__ == "__main__":
    #check if we already have a fine-tuned model
    #if not, train a new model
    #read json file called gpt_models.json
    #if the file exists, load the models
    try:
        with open('gpt_models.json') as json_file:
            gpt_models = json.load(json_file)
    except:
        print("No data file found")
        gpt_models = {}
        
    original_data = f"input_data/{sys.argv[1]}"
    candidate_phrases = f"output_data/{sys.argv[2]}"
    filtered_data = f"output_data/{sys.argv[3]}"
    # finetuned_model = None

    if original_data in gpt_models:
        finetuned_model = gpt_models[original_data]



    #if we have a fine-tuned model, use it to generate counterfactuals

    if finetuned_model:
        generate_counterfactual_withmodel(finetuned_model)

    else:
        
        #train a new model
        fine_tune_model(
            filtered_data,
            candidate_phrases)
        






