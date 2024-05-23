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
import configparser

config = configparser.ConfigParser()
# Read the config.ini file
config.read('config.ini')

API_KEY = config.get("settings", "openai_api").split('#')[0].strip()
DATA_FILE = config.get("settings", "data_file").split('#')[0].strip()
SEED = config.get("settings", "seed").split('#')[0].strip()



#fine-tuned modesl - Yelp => ft:gpt-3.5-turbo-0613:university-of-notre-dame::8US4bHqN
#                   - massive => ft:gpt-3.5-turbo-0613:university-of-notre-dame::8UH39TNK


seed = int(SEED)

client = OpenAI(
    api_key=API_KEY
)

#use the filtered dataset to finetune a smaller GPT model


#use finetuned model to generate counterfactauls
def generate_counterfactual_withmodel(modelname = None):
    count = 0
    print("Generating counterfactuals with the fine-tuned model")
    #generate over candidate phrases and generate counterfactuals 
    df = pd.read_csv(f"output_data/[{seed}]{DATA_FILE[:-4]}_candidate_phrases_annotated_data.csv")
    columns = ["id", "ori_text", "ori_label", "pattern", "highlight", "candidate_phrases", "target_label", "counterfactual"]
    counter_df = pd.DataFrame(columns=columns)
    for index, row in df.iterrows():
        count += 1
        print(f"Processing {index}...")
        text = row["ori_text"]
        label = row["ori_label"]
        target_label = row["target_label"]
        generated_phrases = row["candidate_phrases"]
        hihglight = row["highlight"]
        pattern = row["pattern"]

        test_prompt = f'{text}\nOriginal label: {label}\nPattern: {pattern} \nHighlight: {hihglight} \n Candidate phrases: {generated_phrases} \nTarget label: {target_label}\n\n###\n\n'

        
        response = client.chat.completions.create(
            model=modelname,
            messages=[{"role":"user", "content":test_prompt}]
            )
        counter_df = counter_df.append({"id": row["id"], "ori_text": row["ori_text"], "ori_label": row["ori_label"], "pattern": row["pattern"], "highlight": row["highlight"], "candidate_phrases": row["candidate_phrases"], "target_label": row["target_label"], "counterfactual": response.choices[0].message.content}, ignore_index=True)


    counter_df.to_csv(f"output_data/[{seed}]tuned_counterfactuals_{DATA_FILE[:-4]}.csv", index=False)




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
    
    finetuningdata_path = f"output_data/{DATA_FILE[:-4]}_GPT35_fine_tune_data.jsonl"
   
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
    n_epochs = 1
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

        if status in ["completed", "failed", "succeeded"]:
            #reutrn the fine-tuned model name
            if status == "completed" or status == "succeeded":
                modelname = fine_tuning_status.fine_tuned_model
                print(f"Fine-tuning job completed successfully. Model name: {modelname}")
                return modelname
            else:
                print("Fine-tuning job failed")
                return None
        time.sleep(30)



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
        
    original_data = f"input_data/{DATA_FILE}"
    candidate_phrases = f"output_data/[{SEED}]{DATA_FILE[:-4]}_candidate_phrases_annotated_data.csv"
    filtered_data = f"output_data/[{SEED}]filtered_{DATA_FILE}"


    finetuned_model = None
    if original_data in gpt_models:
        finetuned_model = gpt_models[original_data]

    #if we have a fine-tuned model, use it to generate counterfactuals
    if finetuned_model:
        generate_counterfactual_withmodel(finetuned_model)

    else: #if we don't have a fine-tuned model, train a new model
        #train a new model
        model_name = fine_tune_model(
            filtered_data,
            candidate_phrases)
        if model_name:
            gpt_models[original_data] = model_name
            with open('gpt_models.json', 'w') as outfile:
                json.dump(gpt_models, outfile)
            generate_counterfactual_withmodel(model_name)
        






