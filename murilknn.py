#!/usr/bin/env python
# coding: utf-8

print("Running")
import pandas as pd
import numpy as np
from glob import glob
import re
import json
from tqdm import tqdm
import gc
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
import torch

# Step 1: Gathering directories
directories = []
for i in glob("SentTrans-KNN-ICHCL-main/data/train/*/"):
    for j in glob(i + '*/'):
        directories.append(j)
for i in glob("SentTrans-KNN-ICHCL-main/data/test/*/"):
    for j in glob(i + '*/'):
        directories.append(j)

# Step 2: Loading data and labels
data = []
for i in directories:
    try:
        with open(i + 'data.json', encoding='utf-8') as f:
            data.append(json.load(f))
    except FileNotFoundError:
        continue

labels = []
for i in directories:
    try:
        with open(i + 'labels.json', encoding='utf-8') as f:
            labels.append(json.load(f))
    except FileNotFoundError:
        continue

# Step 3: Loading the fine-tuned MuRIL BERT model and tokenizer
model_path = "fine_tuned_muril"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Step 4: Embedding function using MuRIL BERT
def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token's embedding as the sentence embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding

def abc_ST(d, l):
    embeddings = []
    e = encode_text(d['tweet'])
    embeddings.append({
        'tweet_id': d['tweet_id'],
        'embedding': e,
        'label': l[d['tweet_id']]
    })

    for i in d['comments']:
        c = encode_text(i['tweet'])
        embeddings.append({
            'tweet_id': i['tweet_id'],
            'embedding': 0.1 * e + 0.1 * c,
            'label': l[i['tweet_id']]
        })
        if 'replies' in i.keys():
            for j in i['replies']:
                r = encode_text(j['tweet'])
                embeddings.append({
                    'tweet_id': j['tweet_id'],
                    'embedding': 0.1 * e + 0.1 * c + 0.3 * r,
                    'label': l[j['tweet_id']]
                })
    return embeddings

# Step 5: Populating data_label and handling empty cases
data_label = []

for i in tqdm(range(len(labels))):
    for j in abc_ST(data[i], labels[i]):
        data_label.append(j)

# Step 6: Check data_label is not empty before creating the DataFrame
if data_label:
    try:
        df = pd.DataFrame(data_label, columns=data_label[0].keys(), index=None)
        if df.empty:
            print("DataFrame is empty.")
        else:
            print("DataFrame created successfully. Proceed with further processing.")
    except Exception as e:
        print(f"An error occurred while creating the DataFrame: {e}")
else:
    print("data_label list is empty. Cannot create DataFrame.")

# Step 7: Convert embeddings to numpy arrays for model training
X = [np.array(j) for j in df.embedding]
X = np.array(X)
y = df.label

# Step 8: KFold cross-validation
kf = KFold(n_splits=5, shuffle=True)
accs = []
f1_macros = []

for train_index, test_index in kf.split(X, y):
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X[train_index], y[train_index])
    y_pred = classifier.predict(X[test_index])
    accs.append(float(accuracy_score(y[test_index], y_pred)))
    f1_macros.append(float(f1_score(y[test_index], y_pred, average='macro')))

# Step 9: Training on full dataset
d = {
    'Mean Accuracy': np.mean(np.array(accs)).item(),
    'Mean F1_macro': np.mean(np.array(f1_macros)).item(),
}

classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X[:5740], y[:5740])
y_pred = classifier.predict(X[5740:])
d['Test Accuracy'] = float(accuracy_score(y[5740:], y_pred))
d['Test Macro F1'] = float(f1_score(y[5740:], y_pred, average='macro'))

# Step 10: Save performance metrics as JSON
d = json.dumps(d, indent=4)
print(d)

# Step 11: Calculate distances between positive, negative, and mixed embeddings
pos = []
neg = []
for i in range(len(y)):
    if y[i] == 'HOF':
        pos.append(X[i])
    else:
        neg.append(X[i])

# Positive distance calculation
pos_dist = 0
for i in range(len(pos)):
    for j in range(i, len(pos)):
        pos_dist += np.sqrt(np.sum(np.square(pos[i] - pos[j])))
pos_dist /= (len(pos) * (len(pos) + 1) / 2)

# Negative distance calculation
neg_dist = 0
for i in range(len(neg)):
    for j in range(i, len(neg)):
        neg_dist += np.sqrt(np.sum(np.square(neg[i] - neg[j])))
neg_dist /= (len(neg) * (len(neg) + 1) / 2)

# Positive-Negative distance calculation
pos_neg_dist = 0
for i in range(len(pos)):
    for j in range(len(neg)):
        pos_neg_dist += np.sqrt(np.sum(np.square(pos[i] - neg[j])))
pos_neg_dist /= len(pos) * len(neg)

# Print calculated distances
print(f"Positive distance: {pos_dist}")
print(f"Negative distance: {neg_dist}")
print(f"Positive-Negative distance: {pos_neg_dist}")

joblib.dump(classifier, 'knn_muril_model_fine.pkl')
