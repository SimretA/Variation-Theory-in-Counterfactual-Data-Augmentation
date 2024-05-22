#in this file we use the candidate phrases generated in previous pipeline to generate counterfactuals using gpt4
import sys
import pandas as pd
#import PaTAT things
# from PaTAT_piped.api_helper import APIHelper
import re

from openai import OpenAI


client = OpenAI(
    api_key="<<API_KEY>>"
)
seed=1

if len(sys.argv)<1:
    print("ERROR: No Data file procided.")
    sys.exit(1)

#candidate_phrases_annotated_data_emotions_labeled.csv
df = pd.read_csv(f"output_data/{sys.argv[1]}")

col_names = ["id", "ori_text", "ori_label", "pattern", "highlight", "candidate_phrases", "target_label", "counterfactual"]

data_collector = []


print("LOG: Data loaded successfully. Generating counterfactuals...")
#iterate over row and generate counterfactuals
for index, row in df.iterrows():
    print(f"Processing {index}...")
    text = row["ori_text"]
    label = row["ori_label"]
    target_label = row["target_label"]
    generated_phrases = row["candidate_phrases"]
    hihglight = row["highlight"]
    pattern = row["pattern"]
    
    response2 = client.chat.completions.create(
                # model="gpt-4",
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system" ,"content":"The assistant will create generate a counterfactual example close to the original sentence that contains one of the given phrases."},
                    {"role":"user" ,
                        "content":f'''
                        Your task is to change the given sentence from the current label to the target.
                        For example: 'Find me a train ticket next monday to new york city' with original label:transport would be turned to 'Play me a song called New York City by Taylor Swift' with a label audio.
                        You can use the following phrases to help you generate the counterfactuals.
                        Please make the sentnece about {target_label}. Make sure that the new sentence is not {label}.
                        You must use one of the following phrases without rewording it in the new sentence based on the following three criteria:{generated_phrases}
                        criteria 1: the phrase should change the label from {label} to {target_label} to the highest degree. 
                        criteria 2: the modified sentence can not also be about {label} and make sure the word {target_label} is not part of the modified sentence.
                        criteria 3: the modified sentence should be grammatically correct.
                        '''},
                    {"role":"user", "content":f"original text:{text}, original label:{label}, target label:{target_label}, generated phrases:{generated_phrases}, modified text:"},
                    ],
                    temperature=0, max_tokens=256, stop=["\n"]

            )

    data2 = response2.choices[0].message.content
    data_collector.append([row["id"], text, label, pattern, hihglight, generated_phrases, target_label, data2])

df2 = pd.DataFrame(data_collector, columns=col_names)
df2.to_csv(f"output_data/[{seed}]counterfactuals_{sys.argv[1]}", index=False)