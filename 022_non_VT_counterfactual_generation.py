import sys
import pandas as pd
from openai import OpenAI
import configparser

config = configparser.ConfigParser()
# Read the config.ini file
config.read('config.ini')

API_KEY = config.get("settings", "openai_api").split('#')[0].strip()
DATA_FILE = "massive_540.csv" #config.get("settings", "data_file").split('#')[0].strip()
SEED = config.get("settings", "seed").split('#')[0].strip()

seed = int(SEED)

client = OpenAI(
    api_key=API_KEY
)



#open the candidate phrases file
candidate_file = f"[{SEED}]{DATA_FILE[:-4]}_candidate_phrases_annotated_data.csv"

df = pd.read_csv(f"output_data/{candidate_file}")

col_names = ["id", "ori_text", "ori_label",  "target_label", "counterfactual"]

data_collector = []

print("LOG: Data loaded successfully. Generating non VT counterfactuals...")

#iterate over row and generate counterfactuals
for index, row in df.iterrows():
    if index>12000:
        break
    print(f"Processing {index}...")
    id = row["id"]
    text = row["ori_text"]
    label = row["ori_label"]
    target_label = row["target_label"]
    
    response2 = client.chat.completions.create(
                # model="gpt-4o",
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system" ,"content":"The assistant will generate a counterfactual example close to the original sentence that converts the original label to the provided target label."},
                    {"role":"user" ,
                        "content":f'''
                        Your task is to change the given sentence from the current label to the target.
                        text: 'Find me a train ticket next monday to new york city', original label:'transport', target label:'audio', counterfactual:
                        '''},
                    {"role":"assistant" ,"content":"'Play me a song called New York City by Taylor Swift'"},
                    {"role":"user" ,"content":f"text: {text}, original label:{label}, target label:{target_label}"},

                ]   
            )
    
    data_collector.append([id, text, label, target_label, response2.choices[0].message.content])
    

#save the counterfactuals
df = pd.DataFrame(data_collector, columns=col_names)
df.to_csv(f"output_data/non_VT_counter/[{SEED}]{DATA_FILE[:-4]}_non_VT_counterfactuals.csv", index=False)