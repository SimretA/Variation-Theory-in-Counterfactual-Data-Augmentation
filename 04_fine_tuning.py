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
        highlight = row["highlight"]
        pattern = row["pattern"]

        messages=[{"role": "system", "content": "The assistant will modify a given original text to change its label by making small changes. The modified sentence should be as close to the original sentence as possible. The modified sentence must always include one of the user provided candidate phrases\nThe assistant will modify the given sentence with a goal of changing its current label to the target label while strictly following the following criteria:\ncriteria 1: the modified sentence should change its label from the original label to the target label to the highest degree. However the modified sentence must always include one of the provided candidate phrases. and the assistant will provide which phrase was included in the response\ncriteria 2: the modified sentence can not also be about the original label.\ncriteria 3: the modified sentence should be grammatically correct. The sentence must not contain any contractions. For example, instead of \"I'm\" use \"I am\".\ncriteria 4: the modified sentence should overlap with the original sentence as much as possible. do not make any unnecessary changes or additions to the original sentence. Try to add, change, or remove the least number of words.\ncriteria 5: the modified sentence should not include the literal word of the target label.\n\n"},
                {"role": "user", "content":  "original sentence: 'Find me a train ticket next monday to new york city' , original label:transport, target label: audio, candidate phrases:['sing me a song', 'play me a song', 'show me a train']"},
                {"role": "assistant", "content": "\nmodified sentence: 'Play me a song called New York City by Taylor Swift' "},
                {"role": "user", "content":  "original sentence: \"The wings were delicious .\", original label: product, target label: price, candidate phrases: ['yummy', 'tasty', 'flavour', 'deliciousness', 'taste', 'delicious']\n"},
                {"role": "assistant", "content":"\nmodified sentence: \"The delicious wings were worth every penny.\""},
                {"role": "assistant", "content": "\nmodified sentence: \"The yummy flavor was worth every penny.\""},
                {"role": "user", "content":"original sentence: \"The wings were delicious .\", original label: product, target label: environmnet, candidate phrases: ['January 1st', 'February 14th', 'March 30th', 'April 25th', 'May 10th', 'June 5th', 'July 20th', 'August 15th', 'September 3rd', 'October 12th', 'November 8th', 'December 24th']"},
                {"role": "assistant", "content": "\nmodified sentence: \"The January 1st wings were too cold.\""},
                {"role": "user", "content": "original sentence: \"Too many other places to shop with better prices .\", original label: price, target label: service, candidate phrases: ['costs .', 'pricing .', 'sale .', 'fees .', 'pricing .', '5-$10 .', '40.00 .', 'costs .', 'pricing .']"},
                {"role": "assistant","content": "\nmodified sentence: \"Too many other places to shop with better service costs.\""},
                {"role": "user", "content":  f"original sentence: \"{text}\", original label: {label}, pattern:{pattern}, highlight:{highlight} candidate phrases: {generated_phrases}, target label: {target_label} \n\n###\n\n"},
                 ]

        # test_prompt = f'{text}\nOriginal label: {label}\nPattern: {pattern} \nHighlight: {hihglight} \n Candidate phrases: {generated_phrases} \nTarget label: {target_label}\n\n###\n\n'

        
        response = client.chat.completions.create(
            model=modelname,
            messages=messages,
            temperature=0, max_tokens=256, stop=["\n"]
            )
        # print(f"{text} -- {label}\n\n")

        # print(f"{highlight} -- {pattern}\n\n")
        # print(f"candidate phrases: {generated_phrases}\n\n")

        # print(f"{response.choices[0].message.content} -- {target_label}\n\n")
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
    count = 0

    

    for i, row in data.iterrows():
        print(f"Processing {i}...")

        # check if the the row has gone through the three filters
        passed_filter = row['matched_pattern'] and row['heuristic_filtered'] and row['is_target'] and (not row['is_ori'])

        # id,ori_text,ori_label,pattern,highlight,candidate_phrases,target_label,counterfactual,heuristic_filtered,matched_pattern,is_ori,is_target

        
        if not passed_filter:
            continue
        count += 1

        id = row['id']
        #select a row with the same id from candidate_phrases
        target_label = row['target_label']
        pattern = row['pattern']

        highlight = row['highlight']
        candidate_phrase = row['candidate_phrases']
        try:
            outcome = " "+ row['counterfactual'].strip() + "###"
            message=[{
                    "role": "system", "content": "The assistant will modify a given original text to change its label by making small changes. The modified sentence should be as close to the original sentence as possible. The modified sentence must always include one of the user provided candidate phrases\nThe assistant will modify the given sentence with a goal of changing its current label to the target label while strictly following the following criteria:\ncriteria 1: the modified sentence should change its label from the original label to the target label to the highest degree. However the modified sentence must always include one of the provided candidate phrases. and the assistant will provide which phrase was included in the response\ncriteria 2: the modified sentence can not also be about the original label.\ncriteria 3: the modified sentence should be grammatically correct. The sentence must not contain any contractions. For example, instead of \"I'm\" use \"I am\".\ncriteria 4: the modified sentence should overlap with the original sentence as much as possible. do not make any unnecessary changes or additions to the original sentence. Try to add, change, or remove the least number of words.\ncriteria 5: the modified sentence should not include the literal word of the target label.\n\n"},
                    {"role": "user", "content":  "original sentence: 'Find me a train ticket next monday to new york city' , original label:transport, target label: audio, candidate phrases:['sing me a song', 'play me a song', 'show me a train']"},
                    {"role": "assistant", "content": "\nmodified sentence: 'Play me a song called New York City by Taylor Swift' "},
                    {"role": "user", "content":  "original sentence: \"The wings were delicious .\", original label: product, target label: price, candidate phrases: ['yummy', 'tasty', 'flavour', 'deliciousness', 'taste', 'delicious']\n"},
                    {"role": "assistant", "content":"\nmodified sentence: \"The delicious wings were worth every penny.\""},
                    {"role": "assistant", "content": "\nmodified sentence: \"The yummy flavor was worth every penny.\""},
                    {"role": "user", "content":"original sentence: \"The wings were delicious .\", original label: product, target label: environmnet, candidate phrases: ['January 1st', 'February 14th', 'March 30th', 'April 25th', 'May 10th', 'June 5th', 'July 20th', 'August 15th', 'September 3rd', 'October 12th', 'November 8th', 'December 24th']"},
                    {"role": "assistant", "content": "\nmodified sentence: \"The January 1st wings were too cold.\""},
                    {"role": "user", "content": "original sentence: \"Too many other places to shop with better prices .\", original label: price, target label: service, candidate phrases: ['costs .', 'pricing .', 'sale .', 'fees .', 'pricing .', '5-$10 .', '40.00 .', 'costs .', 'pricing .']"},
                    {"role": "assistant","content": "\nmodified sentence: \"Too many other places to shop with better service costs.\""},
                    {"role": "user", "content":  f"original sentence: \"{data['ori_text'][i].strip()}\", original label: {row['ori_label']}, pattern:{pattern}, highlight:{highlight} candidate phrases: {candidate_phrase}, target label: {target_label} \n\n###\n\n"},
                    {"role":"assistant" ,"content":f"{outcome}"}

                ]
        
            #text =  + '\nOriginal label: ' +  + '\nPattern: ' + pattern +  '\nHighlight: ' + highlight + '\nCandidate phrases: ' + candidate_phrase  + '\nTarget label: ' + target_label+ '\n\n###\n\n'

        except:
            print("Error in processing row")
            continue
        messages.append({"messages":message})
        # prompts.append(text)
        # completions.append(outcome)
        
    #df is the data we will use to finetune a gpt model    
    # df = pd.DataFrame(zip(prompts,completions),columns=['prompt','completion'])
    
    finetuningdata_path = f"output_data/{DATA_FILE[:-4]}_GPT35_fine_tune_data.jsonl"
   
    # df.to_json( finetuningdata_path,orient='records',lines=True)

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
    model_engine = "gpt-3.5-turbo-0125" #"davinci-002"


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
        






