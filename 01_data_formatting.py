import sys
import pandas as pd
#import PaTAT things
from PaTAT_piped.api_helper import APIHelper
import re
import ast
import configparser


from openai import OpenAI


config = configparser.ConfigParser()

# Read the config.ini file
config.read('config.ini')

API_KEY = config.get("settings", "openai_api")



client = OpenAI(
    api_key=API_KEY
)
seed=1

total_annotations = 150
# total_annotations = 50



def get_patat_patterns():
    pass

def get_candidate_phrases():
    pass





print("INFO: Executing data formatting with Python interpreter path:", sys.executable)


#open the data file
if len(sys.argv)<1:
    print("ERROR: No Data file procided.")
    sys.exit(1)

file_path = f"input_data/{sys.argv[1]}"
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


# #get patterns from 5, 10, 15...up to 250 annotations from PaTAT
for i in range(0, len(df), 10):
    #select five random ids at a time and append to annotated_ids
    batch_ids = df.iloc[i:i+5]['id'].to_list()

    #append the selected ids to annotated ids
    annotated_ids.extend(batch_ids)
    if len(annotated_ids)>total_annotations:
        break

    print(f"INFO: Length of annotated ids {len(annotated_ids)} of {total_annotations}")


    #generate patterns using PaTAT
    patat = APIHelper(dataset = df, file_name=sys.argv[1][:-4]) #TODO handle what happens if the cache for similar words can not be found
    #TODO APIHelper should have a flag for caching annotation or not
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
df.to_csv(f"output_data/annotated_data_with_pattern_{sys.argv[1][:-4]}.csv", index=False)

# unique_labels = df["Label"].unique()
patat = APIHelper(dataset = df, file_name=sys.argv[1][:-4])

df = pd.read_csv(f"output_data/annotated_data_with_pattern_{sys.argv[1][:-4]}.csv")

#generate candidate phrases for each sentence and write into a table
#set up data to collect patterns and annotations for the pattern generation. 
col_names_2 = ["id", "ori_text", "ori_label", "pattern", "highlight", "target_label", "candidate_phrases"]
data_collector_2 = []

#table should have ID, setnence, pattern, target_label, candidate phases
for i, row in df.iterrows():
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
        print(f"INFO: Generating candidate phrases for {row['id']} with matched phrase {matched_phrase}")
        #create candidate phrases for each label that is not the origianl label
        for target_label in unique_labels:
            if target_label == label or target_label == 'none':
                continue
            else:
                messages=[
                {"role":"system" ,"content":"The assistant will create a list of phrases that match the given domain specific language based on the given definition."},
                {"role":"user" ,
                    "content":'''The domain specific pattern laguge includes the follwing patterns:
                    Part-of-speech (POS) tags: VERB, PROPN, NOUN, ADJ, ADV, AUX, PRON, NUM
                    Word stemming: [WORD] (e.g., [have] will match all variants of have)
                    Soft match: (word) (word will only be matched with a limited set of similar words provided in this instruction)
                    Entity type: $ENT-TYPE (e.g., $LOCATION will match phrases of location type, such as Houston; $DATE will match dates)
                    Wildcard: * (will match any sequence of words)'''},
                {"role":"user", "content":"The patterns can be combined using an and operator (+) or an or operator (|)."},
                {"role":"user", "content":"For example the pattern 'VERB PROPN' will match any sentence that has a verb followed by a proper noun."},
                {"role":"user", "content":f"For the following text and pattern, generate as many diverse example phrases that match the given pattern and can be part of the given target label. Try to not use the word {label} or {target_label} in the phrases you generate. Separated your answer by a comma"},

                {"role":"user", "content":"text: third bong hit, pattern: $ORDINAL+*+NOUN+VERB, current label:fear, target label:sadness"},
                {"role":"assistant", "content":"'third bong hit', 'third attempt failed', 'first idea works', 'second plan succeeds', 'fourth project launches', 'fifth attempt succeeds', 'sixth strategy works', 'seventh game wins', 'eighth proposal passes', 'ninth scheme fails', 'tenth trial succeeds'"},

                {"role":"user", "content":f"For the following text and pattern, generate as many diverse example phrases that match the given pattern and can be part of the given target label. Try to not use the word {label} or {target_label} in the phrases you generate. Separated your answer by a comma"},
                {"role":"user" ,"content":f"text: {highlight}, pattern: {pattern}, current label: {label} target label: {target_label}"},
                ]
                
                matches = re.findall(r'\(([^)]+)\)', pattern)
                if len(matches)>0:
                    instructions = "Softmatches can only be replaced with a list of available words. "
                    for match in matches:
                        if match in patat.similarity_dict:
                            soft_match_words = list(patat.similarity_dict[match].keys()) + [match]
                            instructions += f"The word `{match}` is a soft match, you can only use {soft_match_words} as its synonyms to replace it. You can not use other words for {match}\n"
                    messages.insert(4,{"role":"user" ,"content":instructions})
                
                response = client.chat.completions.create(
                    # model="gpt-4-1106-preview",
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0, max_tokens=256, stop=["\n"])
                

                data = response.choices[0].message.content
                generated_phrases = [x.strip().replace('"', '').replace("'", "") for x in data.split(",")]
                data_collector_2.append([row["id"], sentence, label, pattern, matched_phrase, target_label, generated_phrases])

df2 = pd.DataFrame(data_collector_2, columns=col_names_2)
df2.to_csv(f"output_data/[{seed}]{sys.argv[1][:-4]}_candidate_phrases_annotated_data.csv", index=False)
