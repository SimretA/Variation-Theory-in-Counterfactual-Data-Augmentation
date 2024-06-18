import sys
import pandas as pd
#import PaTAT things
from PaTAT_piped.api_helper import APIHelper
import re
import ast
import configparser


from openai import OpenAI
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4")


config = configparser.ConfigParser()

# Read the config.ini file
config.read('config.ini')

API_KEY = config.get("settings", "openai_api").split('#')[0].strip()

DATA_FILE = config.get("settings", "data_file").split('#')[0].strip()

SEED = config.get("settings", "seed").split('#')[0].strip()

client = OpenAI(
    api_key=API_KEY
)
seed=int(SEED)

total_annotations = 150
# total_annotations = 50



def get_patat_patterns():
    print("INFO: Executing data formatting with Python interpreter path:", sys.executable)

    #open the data file
    if len(DATA_FILE)<1:
        print("ERROR: No Data file provided.")
        sys.exit(1)

    file_path = f"input_data/{DATA_FILE}"
    try:
        df = pd.read_csv(file_path)
        print("INFO: Finished reading data")
    except:
        print(f"ERROR: can not read file {file_path}")
        sys.exit(1)

    #shuffle the data so that we can randomly select
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    annotated_ids = []
    unique_labels = df["Label"].unique()

    #set up data to collect patterns and annotations for the pattern generation. 
    col_names = ["id", "ori_text", "ori_label", "pattern", "highlight"]
    data_collector = []


    #get patterns from 5, 10, 15...up to 250 annotations from PaTAT
    for i in range(20, len(df), 30):
        #select five random ids at a time and append to annotated_ids
        batch_ids = df.iloc[i:i+5]['id'].to_list()

        #append the selected ids to annotated ids
        annotated_ids.extend(batch_ids)
        if len(annotated_ids)>total_annotations:
            break

        print(f"INFO: Length of annotated ids {len(annotated_ids)} of {total_annotations}")


        #generate patterns using PaTAT
        patat = APIHelper(dataset = df, file_name=DATA_FILE[:-4]) 
        for id in annotated_ids:
            print(f"INFO: labeling {id}, label {df[df['id']==id]['Label'].values[0]}")
            patat.label_element(id, df[df['id']==id]['Label'].values[0])
        
        #generate patterns for each annotated label
        for label in unique_labels:
            patat.set_theme(label)
            resutls = patat.get_linear_model_results() #TODO handle error 
            
            if("explanation" not in resutls):
                print(f"ERROR: Not enough annotations ({len(annotated_ids)}) for {label}, {resutls}")
                continue

            #collect generated patterns into a csv file
            explanations = resutls['explanation']

            #for each pattern check if the selected theme is equal to the original theme
            for pattern_explanation in explanations:
                for sentence_id in explanations[pattern_explanation]:
                    #get the original label
                    original_label = df[df['id']==sentence_id]['Label'].values[0]
                    if label == original_label:
                        data_collector.append([sentence_id, df[df['id']==sentence_id]['example'].values[0], original_label, pattern_explanation, explanations[pattern_explanation][sentence_id]])
            
    # write data collector into csv file
    df = pd.DataFrame(data_collector, columns=col_names)
    df.to_csv(f"output_data/annotated_data_with_pattern_{DATA_FILE[:-4]}.csv", index=False)



def get_candidate_phrases():
    file_path = f"input_data/{DATA_FILE}"
    try:
        df = pd.read_csv(file_path)
        print("INFO: Finished reading data")
    except:
        print(f"ERROR: can not read file {file_path}")
        sys.exit(1)

    unique_labels = df["Label"].unique()
    patat = APIHelper(dataset = df, file_name=DATA_FILE[:-4])
    try:
        df = pd.read_csv(f"output_data/annotated_data_with_pattern_{DATA_FILE[:-4]}.csv")
    except:
        print("ERROR: can not read annotated data file")
        sys.exit(1)
    
    #generate candidate phrases for each sentence and write into a table
    #set up data to collect patterns and annotations for the pattern generation. 
    col_names_2 = ["id", "ori_text", "ori_label", "pattern", "highlight", "target_label", "candidate_phrases"]
    data_collector_2 = []

    num_tokens = 0

    #table should have ID, setnence, pattern, target_label, candidate phases
    for i, row in df.iterrows():
        
        if num_tokens >10000000:
            print("INFO: Skipping the request due to token limit")
            break
        print(f"INFO: Generating candidate phrases for {row['id']}")
        sentence = row['ori_text']
        pattern = row['pattern']
        label = row['ori_label']
        highlight = row['highlight']
        

        try:
            highlight = ast.literal_eval(highlight)
            marked_phrases = [" ".join(h[0]) for h in highlight]
            print(f"INFO: Marked phrases {marked_phrases}")
        except:
            continue

        for matched_phrase in marked_phrases:
            if num_tokens >10000000:
                print("INFO: Skipping the request due to token limit")
                break
            print(f"INFO: Generating candidate phrases for {row['id']} with matched phrase {matched_phrase}")
            #create candidate phrases for each label that is not the origianl label
            for target_label in unique_labels:
                if target_label == label or target_label == 'none':
                    continue
                else:
                    messages=[
                        {"role": "system", "content": [{"text": "The assistant will create a list of candidate phrases that match the given symbolic domain specific pattern. The domain specific pattern definition is given below.\n\nThe domain specific pattern symbols includes the following patterns:\n- Part-of-speech (POS) tags are capital: VERB, PROPN, NOUN, ADJ, ADV, AUX, PRON, NUM\n- Word stemming are surrounded in [] and should have an exact match: [WORD] (e.g., [have] will match all variants of have)\n- Soft match are surrounded by () and will match words with their synonyms. The list of synonms for each soft match in a pattern are given in the user instruction: (word) (word will only be matched with a limited set of similar words provided in this instruction)\n- Entity type start with $ sign: $ENT-TYPE (e.g., $LOCATION will match phrases of location type, such as Houston; $DATE will match dates)\n- Wildcard is the * symbol and can match anything: * (will match any sequence of words)\n\nThe patterns can be combined using an and operator (+) or an or operator (|).\nFor example the pattern 'VERB + PROPN' will match any sentence that has a verb followed by a proper noun.\n\nSoft matches can only be replaced with a list of available words. \n\nFor the following text and pattern, generate as many diverse example phrases that match the given pattern and can be part of the given target label. Separated your answer by a comma",
                            "type": "text"}]},
                        {"role": "user", "content": [{"text": "sentence:'Too many other places to shop with better prices .', phrase to modify: 'prices .', pattern: '(price)+*', current label: price,  softmatch:[price:[purchase, pricey, cheap, cost, pricing]], target label: service,",
                            "type": "text" }]},
                        {"role": "assistant", "content": [{"text": "purchase options, pricey service, cheap help, pricing plans, cost breakdown",
                            "type": "text"}]},
                        {"role": "user", "content": [{"text": "sentence:' they have great produce and seafood', phrase to modify: 'seafood', pattern: '[seafood]|NOUN', current label: products, target label: service\n",
                            "type": "text"
                            }]},
                        {"role": "assistant",
                        "content": [{"text": "hospitality, seafood, help, management, staff",
                            "type": "text"}]},
                        {"role": "user","content": [{"text": "sentence:' the wings were delicious', phrase to modify: 'delicious', pattern: '(delicious)|$DATE', current label: products,  softmatch:[delicious:['taste', 'flavor', 'deliciousness', 'yummy', 'tasty', 'flavour', 'delicious']], target label: price",
                            "type": "text"}]},
                        {"role": "assistant",
                        "content": [{"text": "affordably delicious, on sale today, priced well for their flavour, deliciousness",
                            "type": "text"}]},
                        {"role": "user", "content": [{"text": "sentence:' they should be shut down for terrible service .', phrase to modify: 'service', pattern: '(service)+(manager)+*', current label: service,  softmatch:[service:['customer', 'service'], manager:['management', 'manage', 'manager']], target label: price",
                            "type": "text"}]},
                        {"role": "assistant",
                        "content": [{"text": "[ service charge, service fee, customer cost, manage pricing, management price]",
                            "type": "text"}]}
                    ]
                    
                    req_message = f"sentence:' {sentence}', phrase to modify: '{highlight}', pattern: '{pattern}', current label: {label}, target label: {target_label}"

                    matches = re.findall(r'\(([^)]+)\)', pattern)
        
                    if len(matches)>0:
                        #if we have softmatches open the dictionary and get the list of words
                        instructions = "Softmatches can only be replaced with a list of available words. "
                        softmatch_collector ={}
                        for match in matches:
                            if match in patat.similarity_dict:
                                soft_match_words = list(patat.similarity_dict[match].keys()) + [match]
                                softmatch_collector[match] = soft_match_words
                                instructions += f"The word `{match}` is a soft match, you can only use {soft_match_words} as its synonyms to replace it. You can not use other words for {match}\n"
                        # messages.insert(4,{"role":"user" ,"content":instructions})
                        req_message = f"sentence:' {sentence}', phrase to modify: '{highlight}', pattern: '{pattern}', current label: {label},  softmatch:{softmatch_collector}, target label: {target_label}"
                    messages.append({"role": "system", "content": [{"text": req_message, "type": "text"}]})

                    

                    
                    for message in messages:
                        num_tokens += len(encoding.encode(message["content"][0]["text"]))
                    print(f"INFO: Number of tokens {num_tokens}")
                    
                    # if num_tokens<=6000000:
                    #     print(f"SKIPPING because already generated")
                    #     continue

                    
                    response = client.chat.completions.create(
                        # model="gpt-4o",
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0, max_tokens=256, stop=["\n"])
                    
                    

                    data = response.choices[0].message.content
                    generated_phrases = [x.strip().replace('"', '').replace("'", "") for x in data.split(",")]
                    data_collector_2.append([row["id"], sentence, label, pattern, matched_phrase, target_label, generated_phrases])
    df2 = pd.DataFrame(data_collector_2, columns=col_names_2)
    df2.to_csv(f"output_data/[{seed}]{DATA_FILE[:-4]}_candidate_phrases_annotated_data.csv", index=False)




if __name__ == "__main__":
    get_patat_patterns()
    get_candidate_phrases()











