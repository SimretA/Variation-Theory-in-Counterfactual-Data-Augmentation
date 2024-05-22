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

API_KEY = config.get("settings", "openai_api")

#fine-tuned modesl - Yelp => ft:gpt-3.5-turbo-0613:university-of-notre-dame::7rGMCwdF
#                   - massive => ft:gpt-3.5-turbo-0613:university-of-notre-dame::8UH39TNK


seed = 1

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

def get_response(messages, query, model="gpt-3.5-turbo"):
    query_messages = copy.deepcopy(messages)
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



#generate counterfactuals and get label predictions 


#random
def random_shots(df, df_test, unique_ids ,label_map, N, test_num):
    messages = get_initial_message(list(label_map.values()))
    
    df_unique = df.drop_duplicates(subset='id')

    shuffled_df = df_unique.sample(frac=1, random_state=seed).reset_index(drop=True)

    for i in range(N):
        if i >= len(shuffled_df):
            print("End of data ", i)
            break
        update_example(messages, shuffled_df['ori_text'][i], label_map[shuffled_df['ori_label'][i]])

    
    correct_count = 0
    valid_count = 0
    all_count = 0
    y_true = []
    y_pred = []
    
    df_test_shuffled = df_test.sample(frac=1, random_state=seed).reset_index(drop=True)


    
    for j in range(0, test_num):
        print("Getting  ", j, " of ", test_num)

        if df_test_shuffled['Label'][j] == "none":
            continue
        
        if j >= len(df_test_shuffled):
            print("End of data ", j)
            break
        response = get_response(messages, df_test_shuffled['example'][j])

        y_true.append(label_map[df_test_shuffled['Label'][j]])
        y_pred.append(response)

        if response == label_map[df_test_shuffled['Label'][j]]:
            correct_count = correct_count + 1
        if response in list(label_map.values()):
            valid_count = valid_count + 1
        all_count += 1

    fscore = f1_score(y_true, y_pred, average='macro')

    logger.info(f"Random, {N}, {test_num }, {correct_count}, {valid_count}, {all_count}, {fscore}")
    print("Random: ", N, test_num,  correct_count, valid_count, all_count, fscore)

    #write y_true and y_pred into a csv file
    df = pd.DataFrame(list(zip(y_true, y_pred)), columns =['y_true', 'y_pred'])
    df.to_csv(f"output_data/[{seed}]testrandom_{sys.argv[1][:-4]}_{N}_{test_num}.csv", index=False)


#cluster
def clustered_shots(df, df_test, unique_ids ,label_map, N, test_num):
    selections = [10, 15, 30, 50, 70, 90]
    selected_indices = []

    df_unique = df.drop_duplicates(subset='id')
    #shuffle the data
    df_unique = df_unique.sample(frac=1, random_state=seed).reset_index(drop=True)

    cluster_indices = get_clusters(df_unique)
    results = []
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
        
        df_test_shuffled = df_test.sample(frac=1, random_state=seed).reset_index(drop=True)

        for j in range(0, test_num):
            print("Getting  ", j, " of ", test_num)

            if df_test_shuffled['Label'][j] == "none":
                continue
            
            if j >= len(df_test_shuffled):
                print("End of data ", j)
                break
            response = get_response(messages, df_test_shuffled['example'][j])

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
    df2.to_csv(f"output_data/[{seed}][GPT]_cluster_{sys.argv[1][:-4]}_{N}_{test_num}_prf.csv", index=False)

    

#counterfactuals
def counterfactual_shots(df, df_test, label_map, N, test_num):
    messages = get_initial_message(list(label_map.values()))

    results = []
    
    df_unique = df.drop_duplicates(subset='id')

    shuffled_df = df_unique.sample(frac=1, random_state=seed).reset_index(drop=True)

    for i in range(N):
        if i>= len(shuffled_df):
            print("End of data ", i)
            break
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
            update_example(messages, row['counterfactual'], label_map[row['target_label']])
        # update_example(messages, counters['counterfactual'], label_map[counters['target_label']])
        
    
    #if the model reaches the maximum context length, drop parts of the training set
            
    
    correct_count = 0
    valid_count = 0
    all_count = 0
    y_true = []
    y_pred = []

    df_test_shuffled = df_test.sample(frac=1, random_state=seed).reset_index(drop=True)

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
            break
        response = get_response(messages, df_test_shuffled['example'][j])

        y_true.append(label_map[df_test_shuffled['Label'][j]])
        y_pred.append(response)

        if response == label_map[df_test_shuffled['Label'][j]]:
            correct_count = correct_count + 1
        if response in list(label_map.values()):
            valid_count = valid_count + 1
        all_count += 1

        
    fscore = f1_score(y_true, y_pred, average='macro')
    prf = precision_recall_fscore_support(y_true, y_pred, average='macro')

    # results.append([i, prf[0], prf[1], prf[2]])

    logger.info(f"counter, {N}, {test_num }, {correct_count}, {valid_count}, {all_count}, {fscore}")
    print("counter: ", N, test_num,  correct_count, valid_count, all_count, fscore)

    #write y_true and y_pred into a csv file
    # df = pd.DataFrame(list(zip(y_true, y_pred)), columns =['y_true', 'y_pred'])
    # df.to_csv(f"output_data/[{seed}][GPT]test_counter_{sys.argv[1][:-4]}_{N}_{test_num}.csv", index=False)
    # df2 = pd.DataFrame(results, columns =['shots', 'precision', 'recall', 'fscore'])
    # df2.to_csv(f"output_data/[{seed}][GPT]_counter_{sys.argv[1][:-4]}_{N}_{test_num}_prf.csv", index=False)
    return prf
        
    

        


if __name__ == "__main__":
    results = []
    selections = [10, 15, 30, 50, 70, 90, 120]

    file = f"output_data/{sys.argv[1]}" #filtered_counterfactuals_candidate_phrases_annotated_data_[500]emotions.csv
    test_file = f"input_data/{sys.argv[2]}" #yelplabeled_test.csv
    
    df = pd.read_csv(file)
    df_test = pd.read_csv(test_file)

    train = None
    test = None
    

    #TODO: have a separate file to store the test set

    label_list = df["ori_label"].unique().tolist()
    ids = df["id"].unique().tolist()


    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'AL_results_random_{sys.argv[1][:-4]}.log', encoding='utf-8', level=logging.INFO)

    label_map = {label: f'concept {chr(65 + i)}' for i, label in enumerate(label_list)}

    test_num = 100
    clustered_shots(df, df_test, ids, label_map, 30, test_num)
    # for N in selections:
    #     # random_shots(df, df_test, ids, label_map, N, test_num)
    #     prf = counterfactual_shots(df, df_test, label_map, N, test_num)
    #     results.append([N, prf[0], prf[1], prf[2]])

    # df2 = pd.DataFrame(results, columns =['shots', 'precision', 'recall', 'fscore'])
    # df2.to_csv(f"output_data/[{seed}][GPT]_counter_{sys.argv[1][:-4]}_{N}_{test_num}_prf.csv", index=False)
