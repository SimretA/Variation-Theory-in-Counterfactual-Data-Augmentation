import os
import sys

import pandas as pd

from datasets import Dataset, DatasetDict

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
import numpy as np

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import TensorDataset

from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification

from transformers import AutoTokenizer


from sklearn.metrics import precision_recall_fscore_support

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

import random
from collections import defaultdict
import configparser

config = configparser.ConfigParser()
# Read the config.ini file
config.read('config.ini')

DATA_FILE = config.get("settings", "data_file").split('#')[0].strip()
TEST_FILE = config.get("settings", "test_file").split('#')[0].strip()
SEED = config.get("settings", "seed").split('#')[0].strip()



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"





# Function to encode texts into format suitable for BERT
def encode_texts(texts):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,  
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def counterfactual_case(shuffled_df, df_test_shuffled, seed, data_name="yelp"):

    selections = [10, 15, 30, 50, 70, 90, 120, 150, 170]

    test_size = 100

    results  = []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    for selection in selections:

        torch.cuda.empty_cache()

        label_encoder = LabelEncoder()

        training_text = shuffled_df['ori_text'][:selection]
        training_labels = shuffled_df['ori_label'][:selection]

        print("BEFORE: ", training_text.shape, training_labels.shape)

        #get the counterfactuals that satisfy the training selection
        additional_texts =[]
        additional_labels = []
        for i in range(selection):
            
            id = shuffled_df['id'][i]
            training_df_rows = training_df[training_df['id'] == id]

            for index, row in training_df_rows.iterrows():
                #check if the row passes the filter test
                pass_filter_test = row['matched_pattern'] and row['heuristic_filtered'] and (not row['is_ori']) and row['is_target']
                if pass_filter_test:
                    additional_texts.append(row['counterfactual'])
                    additional_labels.append(row['target_label'])

                    

        additional_texts_series = pd.Series(additional_texts)
        additional_labels_series = pd.Series(additional_labels)

        training_text = training_text.append(additional_texts_series).reset_index(drop=True)
        training_labels = training_labels.append(additional_labels_series).reset_index(drop=True)
        
        print("AFTER: ", training_text.shape, training_labels.shape)
        





        testing_text = df_test_shuffled['example'][:test_size]
        testing_labels = df_test_shuffled['Label'][:test_size]

        # encoding both training and testing labels
        #concatenate the training and testing labels
        label_encoder.fit(training_labels.append(testing_labels))

        num_labels = len(label_encoder.classes_)
        

        input_ids, attention_masks = encode_texts(training_text)

        encoded_training_labels = label_encoder.transform(training_labels)
        encoded_test_labels = label_encoder.transform(testing_labels)

        


        train_dataset = pd.DataFrame({"text":training_text, "label":encoded_training_labels})
        testing_dataset = pd.DataFrame({"text":testing_text, "label":encoded_test_labels})

        train = Dataset.from_pandas(train_dataset)
        test = Dataset.from_pandas(testing_dataset)

        dataset = DatasetDict()
        dataset['train'] = train
        dataset['test'] = test
        tokenized_datasets = dataset.map(tokenize_function, batched=True)


        CLmodel = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels).to(device)

        training_args = TrainingArguments(
            output_dir='./output_data/archive/bert',     
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=5,
            metric_for_best_model = 'eval_loss',
            save_total_limit=6,
            load_best_model_at_end=True,
            no_cuda=True          
        )
        
        trainer = Trainer(
            model=CLmodel,                         
            args=training_args,                  
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['train'],
            # compute_metrics=compute_metrics_multi_class,         
        )
        
        trainer.train()



        # Testing bit
        # Encode the testing texts

        selected_features_dataset = Dataset.from_dict({
            'input_ids': tokenized_datasets['test']['input_ids'],
            'token_type_ids': tokenized_datasets['test']['token_type_ids'],
            'attention_mask': tokenized_datasets['test']['attention_mask']
        })



        # predictions = trainer.predict(tokenized_datasets['test'])
        predictions = trainer.predict(selected_features_dataset)

        probabilities = torch.nn.functional.softmax(torch.tensor(predictions[0]), dim=-1).numpy()

        predicted_labels = np.argmax(probabilities, axis=-1)
        prf = precision_recall_fscore_support(encoded_test_labels, predicted_labels, average='macro')

        results.append([selection, training_text.shape[0], prf[0], prf[1], prf[2]])
    
    
    #write into a csv file
    results_df = pd.DataFrame(results, columns=['selection', "with_counter" ,'precision', 'recall', 'fscore'])
    results_df.to_csv(f'output_data/archive/bert/[{seed}][{data_name}]counter_selection_results.csv', index=False)



def random_case(shuffled_df, df_test_shuffled, seed, data_name):
    selections =  [10, 15, 30, 50, 70, 90, 120, 150, 170]
    test_size = 100

    results  = []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    for selection in selections:

        torch.cuda.empty_cache()

        label_encoder = LabelEncoder()

        training_text = shuffled_df['ori_text'][:selection]
        training_labels = shuffled_df['ori_label'][:selection]

        testing_text = df_test_shuffled['example'][:test_size]
        testing_labels = df_test_shuffled['Label'][:test_size]

        # encoding both training and testing labels
        #concatenate the training and testing labels
        label_encoder.fit(training_labels.append(testing_labels))

        num_labels = len(label_encoder.classes_)
        

        input_ids, attention_masks = encode_texts(training_text)

        encoded_training_labels = label_encoder.transform(training_labels)
        encoded_test_labels = label_encoder.transform(testing_labels)

        


        train_dataset = pd.DataFrame({"text":training_text, "label":encoded_training_labels})
        testing_dataset = pd.DataFrame({"text":testing_text, "label":encoded_test_labels})

        train = Dataset.from_pandas(train_dataset)
        test = Dataset.from_pandas(testing_dataset)

        dataset = DatasetDict()
        dataset['train'] = train
        dataset['test'] = test
        tokenized_datasets = dataset.map(tokenize_function, batched=True)


        CLmodel = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels).to(device)

        training_args = TrainingArguments(
            output_dir='./output_data/archive/bert',     
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=5,
            metric_for_best_model = 'eval_loss',
            save_total_limit=6,
            load_best_model_at_end=True,
            no_cuda=True          
        )
        
        trainer = Trainer(
            model=CLmodel,                         
            args=training_args,                  
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['train'],
            # compute_metrics=compute_metrics_multi_class,         
        )
        
        trainer.train()



        # Testing bit
        # Encode the testing texts

        selected_features_dataset = Dataset.from_dict({
            'input_ids': tokenized_datasets['test']['input_ids'],
            'token_type_ids': tokenized_datasets['test']['token_type_ids'],
            'attention_mask': tokenized_datasets['test']['attention_mask']
        })



        # predictions = trainer.predict(tokenized_datasets['test'])
        predictions = trainer.predict(selected_features_dataset)

        probabilities = torch.nn.functional.softmax(torch.tensor(predictions[0]), dim=-1).numpy()

        predicted_labels = np.argmax(probabilities, axis=-1)
        prf = precision_recall_fscore_support(encoded_test_labels, predicted_labels, average='macro')

        results.append([selection, prf[0], prf[1], prf[2]])

    results_df = pd.DataFrame(results, columns=['selection', 'precision', 'recall', 'fscore'])
    results_df.to_csv(f'output_data/archive/bert/[{seed}][{data_name}]ranodm_selection_results.csv', index=False)


def cluster_case(shuffled_df, df_test_shuffled, seed, data_name):


    #generate data clusters
    model = SentenceTransformer('all-MiniLM-L6-v2')


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    all_text = shuffled_df['ori_text']

    embeddings = model.encode(all_text, convert_to_tensor=True, show_progress_bar=True, device=device)
    embeddings = embeddings.cpu().numpy()
    #reduce the dimensionality of the embeddings
    pca = PCA(n_components=50)
    reduced_embeddings = pca.fit_transform(embeddings)

    #cluster the embeddings
    k=5

    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(reduced_embeddings)
    clusters = kmeans.labels_
    cluster_indices = defaultdict(list)
    for index, cluster_id in enumerate(clusters):
        cluster_indices[cluster_id].append(index)


    #in each iteration annotate one datapoint from a cluster
    selections =  [10, 15, 30, 50, 70, 90]#, 120, 150, 170]
    test_size = 100

    results  = []


    #train model
    selected_indices = []
    for selection in selections:
        
        torch.cuda.empty_cache()

        label_encoder = LabelEncoder()
        
        for cluster_id, indices in cluster_indices.items():
            selected_indices.extend(random.sample(indices, 1))  # Select one from each cluster first
        while len(selected_indices) < selection:
            additional_indices = [index for cluster, indices in cluster_indices.items() for index in indices if index not in selected_indices]
            selected_indices.append(random.choice(additional_indices))
        training_text = []
        training_labels = []

        print(f"Labeled: {selected_indices}")

        for index in selected_indices:
            training_text.append(shuffled_df['ori_text'][index])
            training_labels.append(shuffled_df['ori_label'][index])

        training_labels = pd.Series(training_labels)
        training_text = pd.Series(training_text)
        testing_text = df_test_shuffled['example'][:test_size]
        testing_labels = df_test_shuffled['Label'][:test_size]

        # encoding both training and testing labels
        #concatenate the training and testing labels
        label_encoder.fit(training_labels.append(testing_labels))

        num_labels = len(label_encoder.classes_)
        

        input_ids, attention_masks = encode_texts(training_text)

        encoded_training_labels = label_encoder.transform(training_labels)
        encoded_test_labels = label_encoder.transform(testing_labels)

        


        train_dataset = pd.DataFrame({"text":training_text, "label":encoded_training_labels})
        testing_dataset = pd.DataFrame({"text":testing_text, "label":encoded_test_labels})

        train = Dataset.from_pandas(train_dataset)
        test = Dataset.from_pandas(testing_dataset)

        dataset = DatasetDict()
        dataset['train'] = train
        dataset['test'] = test
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        CLmodel = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=num_labels).to(device)

        training_args = TrainingArguments(
            output_dir='./output_data/archive/bert',     
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=5,
            metric_for_best_model = 'eval_loss',
            save_total_limit=6,
            load_best_model_at_end=True,
            no_cuda=True          
        )
        
        trainer = Trainer(
            model=CLmodel,                         
            args=training_args,                  
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['train'],
            # compute_metrics=compute_metrics_multi_class,         
        )
        
        trainer.train()
        selected_features_dataset = Dataset.from_dict({
            'input_ids': tokenized_datasets['test']['input_ids'],
            'token_type_ids': tokenized_datasets['test']['token_type_ids'],
            'attention_mask': tokenized_datasets['test']['attention_mask']
        })



        # predictions = trainer.predict(tokenized_datasets['test'])
        predictions = trainer.predict(selected_features_dataset)

        probabilities = torch.nn.functional.softmax(torch.tensor(predictions[0]), dim=-1).numpy()

        predicted_labels = np.argmax(probabilities, axis=-1)
        prf = precision_recall_fscore_support(encoded_test_labels, predicted_labels, average='macro')

        results.append([selection, prf[0], prf[1], prf[2]])
    
    results_df = pd.DataFrame(results, columns=['selection', 'precision', 'recall', 'fscore'])
    results_df.to_csv(f'output_data/archive/bert/[{seed}][{data_name}]cluster_selection_results.csv', index=False)





if __name__ == "__main__":


    training_path = f"output_data/[{SEED}]filtered_{DATA_FILE}"
    test_path = f"input_data/{TEST_FILE}"
    dataname = DATA_FILE[:-4]

    seeds = [1, 42, 55, 99, 1234, 92, 765, 555]
    for seed in seeds:
        torch.cuda.empty_cache()

        try:
            #training df
            training_df = pd.read_csv(training_path)
        except:
            print(f"ERROR: can not read file {training_path}")
            sys.exit(1)

        try:
            #testing df
            testing_df = pd.read_csv(test_path)
        except:
            print(f"ERROR: can not read file {test_path}")
            sys.exit(1)

        df_unique = training_df.drop_duplicates(subset='id')

        shuffled_df = df_unique.sample(frac=1, random_state=seed).reset_index(drop=True)

        df_test_shuffled = testing_df.sample(frac=1, random_state=seed).reset_index(drop=True)

        num_labels = len(shuffled_df['ori_label'].unique())

        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        counterfactual_case(shuffled_df, df_test_shuffled, seed, data_name=dataname)
        random_case(shuffled_df, df_test_shuffled, seed, data_name=dataname)
        cluster_case(shuffled_df, df_test_shuffled, seed, data_name=dataname)
        
