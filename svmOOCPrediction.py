!pip install -U sentence-transformers
!pip install ipython-autotime
%load_ext autotime

##Requirement
import torch
import pandas as pd
import random
import numpy as np
import functools
import pickle
import torch.nn as nn
import json
import gdown
# import evaluate
import warnings
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig 
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer,util
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim import AdamW
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

### Path load dataset
train_path = "/kaggle/input/data-unique-img/new_train.csv"
img_path = "/kaggle/input/cosmos/images_test_acm/images_test_acm/test"
test_path = "/kaggle/input/test-new/test_data.json"

class OutContextData(Dataset):
    def __init__(self, data_path: str, image_input_path: str):
        super().__init__()
        self.data_path = data_path
        self.image_input_path = image_input_path
        self.__load_file()

    def __load_file(self):
        df = pd.read_csv(self.data_path)
        self.__raw_data = df.to_dict('records')
        self.__raw_len = len(self.__raw_data)

    def __len__(self) -> int:
        return self.__raw_len

    def __getitem__(self, idx):
        # Access data based on index
        d_point = self.__raw_data[idx]
        
        caption1 = d_point['caption1']
        caption2 = d_point['caption2']
        
        label = int(d_point['label'])
        
        return {
            "caption1": caption1,
            "caption2": caption2,
            "label": label
        }
    
class LoadTest(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.data = [json.loads(line) for line in file]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'img_local_path': item.get('img_local_path'),
            'caption1': item.get('caption1'),
            'caption2': item.get('caption2'),
            'label': item.get('context_label'),
        }
class ExplainableModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert_config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.intermediate = AutoModel.from_pretrained(model_name)

        output_size = 2 * self.bert_config.hidden_size
        self.output = nn.Linear(output_size, output_size)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output  
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        mean_embeddings = sum_embeddings / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return mean_embeddings

    def forward(self, input_ids_tuple, attention_mask_tuple):
        input_ids1, attention_mask1 = input_ids_tuple
        input_ids2, attention_mask2 = attention_mask_tuple

        with torch.no_grad():
            model_output1 = self.intermediate(input_ids1, attention_mask=attention_mask1).last_hidden_state
            model_output2 = self.intermediate(input_ids2, attention_mask=attention_mask2).last_hidden_state

        sentence_embeddings1 = self.mean_pooling(model_output1, attention_mask1)
        sentence_embeddings1 = F.normalize(sentence_embeddings1, p=2, dim=1)

        sentence_embeddings2 = self.mean_pooling(model_output2, attention_mask2)
        sentence_embeddings2 = F.normalize(sentence_embeddings2, p=2, dim=1)

        return sentence_embeddings1, sentence_embeddings2

def get_ids(text):
    caption1_str = text['caption1']
    caption2_str = text['caption2']

    inputs1 = tokenizer(
        caption1_str,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )

    inputs2 = tokenizer(
        caption2_str,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )

    return inputs1, inputs2

def get_embeddings(data_loader):
    all_embeddings_caption1 = []
    all_embeddings_caption2 = []
    all_labels = []

    for batch in data_loader:
        inputs1_tensor_list = []
        inputs2_tensor_list = []

        labels_batch = batch['label']
        valid_indices = [i for i, label in enumerate(labels_batch) if label is not None]
        labels_batch = [label for label in labels_batch if label is not None]
        labels_tensor = torch.tensor(labels_batch, dtype=torch.float32).to('cuda')

        inputs1, inputs2 = get_ids(batch)
        inputs1_tensor_list.append(inputs1.to('cuda'))
        inputs2_tensor_list.append(inputs2.to('cuda'))

        processed_batch = {
            'ids1': inputs1_tensor_list,
            'ids2': inputs2_tensor_list,
            'labels': labels_tensor,
        }

        embeddings_caption1, embeddings_caption2 = explainable_model(
            (processed_batch['ids1'][0]['input_ids'], processed_batch['ids1'][0]['attention_mask']),
            (processed_batch['ids2'][0]['input_ids'], processed_batch['ids2'][0]['attention_mask'])
        )

        embeddings_caption1 = embeddings_caption1.cpu().numpy()
        embeddings_caption2 = embeddings_caption2.cpu().numpy()
        labels = labels_tensor.cpu().numpy()

        all_embeddings_caption1.append(embeddings_caption1)
        all_embeddings_caption2.append(embeddings_caption2)
        all_labels.append(labels)

    all_embeddings_caption1 = np.concatenate(all_embeddings_caption1, axis=0)
    all_embeddings_caption2 = np.concatenate(all_embeddings_caption2, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_embeddings_caption1, all_embeddings_caption2, all_labels

class HeuristicDataLoader(Dataset):
    def __init__(self, df):
        self.data = df.to_dict('records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'img_local_path': item.get('img_local_path'),
            'caption1': item.get('caption1'),
            'caption2': item.get('caption2'),
            'label': item.get('true_label')
        }
    
def get_embeddings_nli(texts, model, tokenizer, device):
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        embeddings = torch.nn.functional.softmax(logits, dim=1)

    return embeddings.cpu().numpy()


def collate_fn_combined(batch, model=None, tokenizer=None, device=None):
    nli_true_postfix = " is true"
    similarities = []
    nli_scores_true = []

    premise_true = [f'{item["caption1"]}{nli_true_postfix}' for item in batch]
    hypothesis = [item['caption2'] for item in batch]

    embeddings_caption1 = get_embeddings_nli(premise_true, model, tokenizer, device)
    embeddings_caption2 = get_embeddings_nli(premise_true, model, tokenizer, device)

    if model is not None and tokenizer is not None and device is not None:
        inputs_true = tokenizer(premise_true, hypothesis, truncation=True, padding=True, return_tensors='pt')
        inputs_true = {key: tensor.to(device) for key, tensor in inputs_true.items()}

        with torch.no_grad():
            output_true = model(**inputs_true)
            predictions_true = torch.softmax(output_true.logits, dim=-1)
            prediction_probs_true = predictions_true.cpu().numpy()
            
            for pred_true in tqdm(prediction_probs_true, desc="Calculating NLI scores", total=len(batch)):
                label_names = ["entailment", "neutral", "contradiction"]
                prediction_true = {name: round(float(prob) * 100, 1) for prob, name in zip(pred_true, label_names)}
                nli_scores_true.append(prediction_true['contradiction'])

    for idx, item in enumerate(batch):
        item['nli_score_is_true'] = nli_scores_true[idx]
        item['simscore'] = cos_sim(embeddings_caption1[idx], embeddings_caption2[idx]).item()

    return batch

class Prepare_data_pred(Dataset):
    def __init__(self, df):
        self.data = df.to_dict('records')
        self.nli = 0.01 #0.75

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = {'nli_score_is_true': item.get('nli_score_is_true')}

        prediction_info = {'predict': None}
        first_sen_contrast = x['nli_score_is_true'] >= self.nli
        if first_sen_contrast:
            prediction_info['predict'] = 1  
        else:
            prediction_info['predict'] = 0 

        return {**x, **prediction_info}
    
    
model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

explainable_model = ExplainableModel(model_name)
explainable_model = explainable_model.to('cuda')  # Move 
explainable_model.eval()
batch_size = 64
train_dataset = OutContextData(train_path, img_path)
test_data = LoadTest(test_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)


train_embeddings_caption1, train_embeddings_caption2, train_labels = get_embeddings(train_loader)
test_embeddings_caption1, test_embeddings_caption2, test_labels = get_embeddings(test_loader)

train_embeddings_combined = np.concatenate([train_embeddings_caption1, train_embeddings_caption2], axis=1)
test_embeddings_combined = np.concatenate([test_embeddings_caption1, test_embeddings_caption2], axis=1)

train_labels = torch.from_numpy(train_labels)
test_labels = torch.from_numpy(test_labels)

##SVM


svm_params = {'C': 1}
svm_model = LinearSVC(**svm_params)
svm_model.fit(train_embeddings_combined, train_labels)
y_pred = svm_model.predict(test_embeddings_combined)
# accuracy_step1 = accuracy_score(test_labels.numpy(), y_pred)
# print(f"Accuracy on the test set: {accuracy_step1}")

results = []
for i, y_pred in enumerate(y_pred):
    item = test_loader.dataset[i]

    result = {
        'img_local_path': item['img_local_path'],
        'caption1': item['caption1'],
        'caption2': item['caption2'],
        'true_label': item['label'],
        'y_pred': y_pred
    }

    results.append(result)

df = pd.DataFrame(results)
df1 = df[df['y_pred'] == 0.0].copy()

dataset = HeuristicDataLoader(df)
df_loader = DataLoader(dataset, batch_size=64)
embeddings_caption1, embeddings_caption2, labels = get_embeddings(df_loader)
cosine_similarities = cosine_similarity(embeddings_caption1, embeddings_caption2)
df['cosine_similarity'] = cosine_similarities.diagonal()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.eval()

data_loader = DataLoader(dataset, batch_size=64, collate_fn=lambda batch: collate_fn_combined(batch, model=model, tokenizer=tokenizer, device=device))
dataframes = []

for i, batch in enumerate(data_loader):
    df1 = pd.DataFrame(batch)
    dataframes.append(df1)

merged_df = pd.concat(dataframes, ignore_index=True)


    
test_set = Prepare_data_pred(merged_df)
df_new_data = pd.DataFrame([test_set[idx] for idx in range(len(test_set))])
df_original = merged_df.copy()
df_updated = pd.concat([df_original, df_new_data], axis=1)
df['new_y_pred'] = df_updated['predict'].astype(float)

df.loc[((df['cosine_similarity'] > 0.5) & (df_updated['predict'] == 1)) & (df['y_pred'] == 0), 'new_y_pred'] = 0

actual_labels = df['true_label'].values
predicted_labels = df['new_y_pred'].values

correct_predictions = (actual_labels == predicted_labels).sum()
total_predictions = len(df)

accuracy = (correct_predictions / total_predictions)
recall = recall_score(actual_labels, predicted_labels)
precision = precision_score(actual_labels, predicted_labels)
f1 = f1_score(actual_labels, predicted_labels)


# Print  metrics
print(f"Accuracy: {accuracy:.4f}%")
print(f"Recall: {recall:.4f}%")
print(f"Precision: {precision:.4f}%")
print(f"F1 Score: {f1:.4f}%")