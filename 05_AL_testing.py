import logging

import random
import pandas as pd
import sys
import copy
from sklearn.metrics import f1_score

from openai import OpenAI
import tiktoken

from sklearn.metrics import precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from collections import defaultdict
import torch


import configparser

config = configparser.ConfigParser()
# Read the config.ini file
config.read('config.ini')

DATA_FILE = config.get("settings", "data_file").split('#')[0].strip()
TEST_FILE = config.get("settings", "test_file").split('#')[0].strip()
SEED = config.get("settings", "seed").split('#')[0].strip()
API_KEY = config.get("settings", "openai_api").split('#')[0].strip()


seed = int(SEED)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


client = OpenAI(
    api_key=API_KEY
)

def get_clusters(shuffled_df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_text = shuffled_df['ori_text'].tolist()

    embeddings = model.encode(all_text, convert_to_tensor=True, show_progress_bar=True, device=device)
    
    embeddings = embeddings.cpu().numpy()
    #reduce the dimensionality of the embeddings
    pca = PCA(n_components=50)
    reduced_embeddings = pca.fit_transform(embeddings)

    k=5
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(reduced_embeddings)
    clusters = kmeans.labels_
    cluster_indices = defaultdict(list)
    for index, cluster_id in enumerate(clusters):
        cluster_indices[cluster_id].append(index)

    return cluster_indices


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_initial_message(label_list):
    messages=[
            {"role": "system", "content": f"You will predict one label for a text. All the labels have been masked, and you need to learn the pattern from given examples"},
            {"role": "system", "content": f"You must choose label from the following list: {label_list}"},
            {"role": "user", "content": "the doctor does not seem to be very knowledgeable, their treatment did not work."},
            {"role": "assistant", "content": "concept B"}
        ]
    return messages

def get_initial_message_with_confidence(label_list):
    messages=[
            {"role": "system", "content": f"You will predict one label for a text. All the labels have been masked, and you need to learn the pattern from given examples"},
            {"role": "system", "content": f"You must choose label from the following list: {label_list}"},
            {"role": "user", "content": "the doctor does not seem to be very knowledgeable, their treatment did not work."},
            {"role": "assistant", "content": "concept B, confidence: 0.806"}
        ]
    return messages

def get_response(messages, query, model="gpt-3.5-turbo"):
    query_messages = copy.deepcopy(messages)
    query_messages.append({"role": "user", "content": query})
    
    response = client.chat.completions.create(
    model=model,
    messages=query_messages,
    temperature=0, max_tokens=256, stop=["\n"]
    )
    return response.choices[0].message.content



def get_response_with_confidence(messages, query, model="gpt-3.5-turbo"):
    query_messages = copy.deepcopy(messages)
    query_messages.append({"role": "user", "content": "return the confidence of the prediction along with the prediction"})
    query_messages.append({"role": "user", "content": query})
    
    response = client.chat.completions.create(
    model=model,
    messages=query_messages,
    temperature=0, max_tokens=256, stop=["\n"]
    )
    return response.choices[0].message.content


def update_example(messages, text, label):
    messages.append({"role": "user", "content": text})
    messages.append({"role": "assistant", "content": label})


def update_example_with_confidence(messages, text, label):
    random_conf = random.uniform(0.1, 1) 

    random_conf = round(random_conf, 3)

    messages.append({"role": "user", "content": text})
    messages.append({"role": "assistant", "content": f"{label}, confidence: {random_conf}"})




#random
def random_shots(df, df_test, unique_ids ,label_map, N, test_num, shuffleSeed):
    selections = [10, 15, 30, 50, 70, 90, 120]

    
    
    df_unique = df.drop_duplicates(subset='id')

    shuffled_df = df_unique.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)
    
    for selection in selections:
        messages = get_initial_message(list(label_map.values()))
        
        for i in range(selection):
            if i >= len(shuffled_df):
                print("End of data ", i)
                break
            update_example(messages, shuffled_df['ori_text'][i], label_map[shuffled_df['ori_label'][i]])

    
        correct_count = 0
        valid_count = 0
        all_count = 0
        y_true = []
        y_pred = []
        
        df_test_shuffled = df_test.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)


    
        for j in range(0, test_num):
            print("Getting  ", j, " of ", test_num)

            if df_test_shuffled['Label'][j] == "none":
                continue
            
            if j >= len(df_test_shuffled):
                print("End of data ", j)
                break
            response = get_response(messages, df_test_shuffled['example'][j])
            if df_test_shuffled['Label'][j] not in label_map:
                continue
            y_true.append(label_map[df_test_shuffled['Label'][j]])
            y_pred.append(response)

            if response == label_map[df_test_shuffled['Label'][j]]:
                correct_count = correct_count + 1
            if response in list(label_map.values()):
                valid_count = valid_count + 1
            all_count += 1

        fscore = f1_score(y_true, y_pred, average='macro')
        prf = precision_recall_fscore_support(y_true, y_pred, average='macro')
        results.append([selection, prf[0], prf[1], prf[2]])

        logger.info(f"Random, {selection}, {test_num }, {correct_count}, {valid_count}, {all_count}, {fscore}")
        print("Random: ", selection, test_num,  correct_count, valid_count, all_count, fscore)

    #write y_true and y_pred into a csv file
    df = pd.DataFrame(results, columns =['shots', 'precision', 'recall', 'fscore'])
    df.to_csv(f"output_data/archive/gpt/[{shuffleseed}][GPT]random_{DATA_FILE[:-4]}.csv", index=False)

def uncertainty_shots(df, df_test, label_map, N, test_num, shuffleSeed):
    selections = [10, 15, 30, 50, 70, 90, 120]
    selected_ids = []
    results = []

    df_unique = df.drop_duplicates(subset='id')
    #shuffle the data
    df_unique = df_unique.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)

    reamining_df = df_unique

    uncertainty_indices = []
    

    for index, selection in enumerate(selections):
        messages = get_initial_message_with_confidence(list(label_map.values()))
        #if it is the first iteration, select random examples
        if index == 0:
            #randomly select indicise from df_unique
            selected_ids.extend(df_unique['id'][:selection].tolist())

            for idx in selected_ids:
                text = df_unique[df_unique['id'] == idx]['ori_text'].values[0]
                label = df_unique[df_unique['id'] == idx]['ori_label'].values[0]

                #update examples with confidence
                update_example_with_confidence(messages, text, label_map[label])
        else:
            #otherwise select the remainint examples from the uncertainity indices
            annotation_count = selection - len(selected_ids)
            top_uncertainty = uncertainty_indices[:annotation_count]
            # print("LOGGING REMAINING ", reamining_df.iloc[top_uncertainty]['id'].values)

            for uncertainity_idx in top_uncertainty: 
                selected_ids.append(reamining_df['id'].iloc[uncertainity_idx])
                text = reamining_df['ori_text'].iloc[uncertainity_idx]
                label = reamining_df['ori_label'].iloc[uncertainity_idx]

                #update examples with confidence
                update_example_with_confidence(messages, text, label_map[label])
                

        # now get the uncertainity for the remaining examples in remaining_df
        reamining_df = reamining_df[~reamining_df['id'].isin(selected_ids)]
        probabilities = []

        for i in range(len(reamining_df)):

            text = reamining_df['ori_text'].iloc[i]
            response = get_response_with_confidence(messages, text)
            # print(f"RESPONSE: {response}")
            try:
                response_label = response.split(",")[0]
                response_confidence = float(response.split(",")[1].split(":")[1])
            except:
                response_label = response
                response_confidence = 0.5
            
            probabilities.append(float(response_confidence))
        #change probabilities to numpy array
        probabilities = np.array(probabilities)
        uncertainties = 1 - probabilities
        uncertainty_indices = np.argsort(uncertainties)[::-1]


        all_count = 0
        y_true = []
        y_pred = []
        
        df_test_shuffled = df_test.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)

        for j in range(0, test_num):
            print("Getting  ", j, " of ", test_num)

            if df_test_shuffled['Label'][j] == "none":
                continue
            
            if j >= len(df_test_shuffled):
                print("End of data ", j)
                break
            response = get_response_with_confidence(messages, df_test_shuffled['example'][j])
            try:
                response_label = response.split(",")[0]
                response_confidence = response.split(",")[1].split(":")[1]
            except:
                response_label = response
                response_confidence = 0.5

            print(f"ID:{df_test_shuffled['id'][j]}, Label: {response_label}, Confidence: {response_confidence}")
            
            #if label is not in label_map, skip
            if df_test_shuffled['Label'][j] not in label_map:
                continue
            y_true.append(label_map[df_test_shuffled['Label'][j]])
            response_label = label_map[df_test_shuffled['Label'][j]] if label_map[df_test_shuffled['Label'][j]] in response_label else response_label
            y_pred.append(response_label)

            all_count += 1
            
        prf = precision_recall_fscore_support(y_true, y_pred, average='macro')
        results.append([selection, prf[0], prf[1], prf[2]])

        
    df2 = pd.DataFrame(results, columns =['shots', 'precision', 'recall', 'fscore'])
    df2.to_csv(f"output_data/archive/gpt/[{shuffleSeed}][GPT]_uncertainty_{DATA_FILE[:-4]}_prf.csv", index=False)




#cluster
def clustered_shots(df, df_test, unique_ids ,label_map, N, test_num, shuffleSeed):
    selections = [10, 15, 30, 50, 70, 90, 120]
    selected_indices = []
    results = []

    df_unique = df.drop_duplicates(subset='id')
    #shuffle the data
    df_unique = df_unique.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)

    cluster_indices = get_clusters(df_unique)

    for selection in selections:
        messages = get_initial_message(list(label_map.values()))
        cluster_indices = get_clusters(df)
        for cluster_id, indices in cluster_indices.items():
            selected_indices.extend(random.sample(indices, 1))
        while len(selected_indices) < selection:
            additional_indices = [index for cluster, indices in cluster_indices.items() for index in indices if index not in selected_indices]
            selected_indices.append(random.choice(additional_indices))
        for i in selected_indices:
            update_example(messages, df['ori_text'][i], label_map[df['ori_label'][i]])

        correct_count = 0
        valid_count = 0
        all_count = 0
        y_true = []
        y_pred = []
        
        df_test_shuffled = df_test.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)

        for j in range(0, test_num):
            print("Getting  ", j, " of ", test_num)

            if df_test_shuffled['Label'][j] == "none":
                continue
            
            if j >= len(df_test_shuffled):
                print("End of data ", j)
                break
            response = get_response(messages, df_test_shuffled['example'][j])
            #if label is not in label_map, skip
            if df_test_shuffled['Label'][j] not in label_map:
                continue
            y_true.append(label_map[df_test_shuffled['Label'][j]])
            y_pred.append(response)

            if response == label_map[df_test_shuffled['Label'][j]]:
                correct_count = correct_count + 1
            if response in list(label_map.values()):
                valid_count = valid_count + 1
            all_count += 1
        prf = precision_recall_fscore_support(y_true, y_pred, average='macro')
        results.append([selection, prf[0], prf[1], prf[2]])
    df2 = pd.DataFrame(results, columns =['shots', 'precision', 'recall', 'fscore'])
    df2.to_csv(f"output_data/archive/gpt/[{shuffleSeed}][GPT]_cluster_{DATA_FILE[:-4]}_prf.csv", index=False)

    

#counterfactuals
def counterfactual_shots(df, df_test, label_map, N, test_num, shuffleSeed):
    selections = [10, 15, 30, 50, 70, 90, 120]

    
    
    
    df_unique = df.drop_duplicates(subset='id')

    shuffled_df = df_unique.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)
    results = []
    for selection in selections:
        
        messages = get_initial_message(list(label_map.values()))

        for i in range(selection):
            if i>= len(shuffled_df):
                print("End of data ", i)
                continue
            update_example(messages, shuffled_df['ori_text'][i], label_map[shuffled_df['ori_label'][i]])

            # get counterfactuals for all the selected examples
            current_id = shuffled_df['id'][i]
            counters = df[(df['id']==current_id) 
                        & (df['matched_pattern']) 
                        & (df['heuristic_filtered'])
                        & ( ~df['is_ori'])
                        & (df['is_target'])]
            #if counters is more than 4, select 4 random counterfactuals from the list
            if counters.shape[0] > 4:
                #select 4 random counterfactuals from the list
                counters = counters.sample(n=4, random_state=seed).reset_index(drop=True)
                # print(f"ADDING - {counters.shape} - COUNTERFACTUALS ")
            for index, row in counters.iterrows():
                if row['target_label'] not in label_map:
                    continue
                update_example(messages, row['counterfactual'], label_map[row['target_label']])
            # update_example(messages, counters['counterfactual'], label_map[counters['target_label']])
        
    
        correct_count = 0
        valid_count = 0
        all_count = 0
        y_true = []
        y_pred = []

        df_test_shuffled = df_test.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)

        for j in range(0, test_num):
            print("Getting  ", j, " of ", test_num)

            if df_test_shuffled['Label'][j] == "none":
                continue
            
            if j >= len(df_test_shuffled):
                print("End of data ", j)
                break
            tokens = num_tokens_from_string(str(messages))
            if tokens > 16385:
                print(f"End of tokens {tokens}, examples {j}")
                
            try:
                response = get_response(messages, df_test_shuffled['example'][j])
            except:
                print("ERROR: ", j)
                continue

            if df_test_shuffled['Label'][j] not in label_map:
                print("Label not in label map ", df_test_shuffled['Label'][j])
                continue
            y_true.append(label_map[df_test_shuffled['Label'][j]])
            y_pred.append(response)

            if response == label_map[df_test_shuffled['Label'][j]]:
                correct_count = correct_count + 1
            if response in list(label_map.values()):
                valid_count = valid_count + 1

            all_count += 1

            
        # fscore = f1_score(y_true, y_pred, average='macro')
        print(f"Y_TRUE: {y_true}")
        print(f"Y_PRED: {y_pred}")
        prf = precision_recall_fscore_support(y_true, y_pred, average='macro')

        results.append([selection, prf[0], prf[1], prf[2]])

    # logger.info(f"counter, {N}, {test_num }, {correct_count}, {valid_count}, {all_count}, {fscore}")
    # print("counter: ", N, test_num,  correct_count, valid_count, all_count, fscore)

    #write y_true and y_pred into a csv file
    # df = pd.DataFrame(list(zip(y_true, y_pred)), columns =['y_true', 'y_pred'])
    # df.to_csv(f"output_data/[{seed}][GPT]test_counter_{sys.argv[1][:-4]}_{N}_{test_num}.csv", index=False)
    df2 = pd.DataFrame(results, columns =['shots', 'precision', 'recall', 'fscore'])
    df2.to_csv(f"output_data/archive/gpt/[{shuffleSeed}][GPT]_counter_{DATA_FILE[:-4]}_prf.csv", index=False)
    return prf
        
    
#Non Variation Theory counterfactuals
def non_VTcounterfactual_shots(df, df_test, label_map, N, test_num, shuffleSeed):
    selections = [10, 15, 30, 50, 70, 90, 120]

    
    
    
    df_unique = df.drop_duplicates(subset='id')

    shuffled_df = df_unique.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)
    results = []
    for selection in selections:
        
        messages = get_initial_message(list(label_map.values()))

        for i in range(selection):
            if i>= len(shuffled_df):
                print("End of data ", i)
                continue
            update_example(messages, shuffled_df['ori_text'][i], label_map[shuffled_df['ori_label'][i]])

            # get counterfactuals for all the selected examples
            current_id = shuffled_df['id'][i]
            counters = df[(df['id']==current_id) ]
            #if counters is more than 4, select 4 random counterfactuals from the list
            if counters.shape[0] > 4:
                #select 4 random counterfactuals from the list
                counters = counters.sample(n=4, random_state=seed).reset_index(drop=True)
                # print(f"ADDING - {counters.shape} - COUNTERFACTUALS ")
            for index, row in counters.iterrows():
                if row['target_label'] not in label_map:
                    continue
                update_example(messages, row['counterfactual'], label_map[row['target_label']])
            # update_example(messages, counters['counterfactual'], label_map[counters['target_label']])
        
    
        correct_count = 0
        valid_count = 0
        all_count = 0
        y_true = []
        y_pred = []

        df_test_shuffled = df_test.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)

        for j in range(0, test_num):
            print("Getting  ", j, " of ", test_num)

            if df_test_shuffled['Label'][j] == "none":
                continue
            
            if j >= len(df_test_shuffled):
                print("End of data ", j)
                break
            tokens = num_tokens_from_string(str(messages))
            if tokens > 16385:
                print(f"End of tokens {tokens}, examples {i}")

            try:
                response = get_response(messages, df_test_shuffled['example'][j])
            except:
                print("ERROR: ", j)
                continue

            if df_test_shuffled['Label'][j] not in label_map:
                continue
            y_true.append(label_map[df_test_shuffled['Label'][j]])
            y_pred.append(response)

            if response == label_map[df_test_shuffled['Label'][j]]:
                correct_count = correct_count + 1
            if response in list(label_map.values()):
                valid_count = valid_count + 1
            all_count += 1

            
        fscore = f1_score(y_true, y_pred, average='macro')
        prf = precision_recall_fscore_support(y_true, y_pred, average='macro')

        results.append([selection, prf[0], prf[1], prf[2]])

    # logger.info(f"counter, {N}, {test_num }, {correct_count}, {valid_count}, {all_count}, {fscore}")
    # print("counter: ", N, test_num,  correct_count, valid_count, all_count, fscore)

    #write y_true and y_pred into a csv file
    # df = pd.DataFrame(list(zip(y_true, y_pred)), columns =['y_true', 'y_pred'])
    # df.to_csv(f"output_data/[{seed}][GPT]test_counter_{sys.argv[1][:-4]}_{N}_{test_num}.csv", index=False)
    df2 = pd.DataFrame(results, columns =['shots', 'precision', 'recall', 'fscore'])
    df2.to_csv(f"output_data/archive/gpt/nonVT[{shuffleSeed}][GPT]_counter_{DATA_FILE[:-4]}_prf.csv", index=False)
    # return prf
        
    

        


if __name__ == "__main__":
    results = []
    selections = [10, 15, 30, 50, 70, 90, 120]
    seedss = [1, 42, 55, 92, 99, 555, 765, 1234] #[1, 42, 55, 92, 99, 555, 765, 1234] 

    training_path = f"output_data/[{SEED}]filtered_{DATA_FILE}"
    test_path = f"input_data/{TEST_FILE}"
    nonVT_path = f"output_data/non_VT_counter/[{SEED}]{DATA_FILE[:-4]}_non_VT_counterfactuals.csv"
    
    try:
        #training df
        df = pd.read_csv(training_path)
    except:
        print(f"ERROR: can not read file {training_path}")
        sys.exit(1)

    try:
        #testing df
        df_test = pd.read_csv(test_path)
    except:
        print(f"ERROR: can not read file {test_path}")
        sys.exit(1)
    try:
        #non VT df
        non_VT_df = pd.read_csv(nonVT_path)
    except:
        print(f"ERROR: can not read file {nonVT_path}")
        sys.exit(1)

    
    label_list = df["ori_label"].unique().tolist()
    ids = df["id"].unique().tolist()


    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'AL_results_random_{DATA_FILE[:-4]}.log', encoding='utf-8', level=logging.INFO)

    label_map = {label: f'concept {chr(65 + i)}' for i, label in enumerate(label_list)}

    test_num = 100
    for shuffleseed in seedss:
        clustered_shots(df, df_test, ids, label_map, 30, test_num, shuffleSeed=shuffleseed)
        random_shots(df, df_test, ids, label_map, 30, test_num, shuffleSeed=shuffleseed)
        counterfactual_shots(df, df_test, label_map, 30, test_num, shuffleSeed=shuffleseed)
        uncertainty_shots(df, df_test, label_map, 30, test_num, shuffleSeed=shuffleseed)
        non_VTcounterfactual_shots(non_VT_df, df_test, label_map, 30, test_num, shuffleSeed=shuffleseed)
        
