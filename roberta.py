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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

# Step 3: Load the Cardiff NLP model and tokenizer
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Step 4: Embedding function using Cardiff NLP model
def get_embeddings(d, l):
    embeddings = []
    e = get_model_embedding(d['tweet'])
    embeddings.append({
        'tweet_id': d['tweet_id'],
        'embedding': e,
        'label': l[d['tweet_id']]
    })

    for i in d['comments']:
        c = get_model_embedding(i['tweet'])
        embeddings.append({
            'tweet_id': i['tweet_id'],
            'embedding': 0.1 * e + 0.1 * c,
            'label': l[i['tweet_id']]
        })
        if 'replies' in i.keys():
            for j in i['replies']:
                r = get_model_embedding(j['tweet'])
                embeddings.append({
                    'tweet_id': j['tweet_id'],
                    'embedding': 0.1 * e + 0.1 * c + 0.3 * r,
                    'label': l[j['tweet_id']]
                })
    return embeddings

def get_model_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.logits.detach().numpy().flatten()

# Step 5: Populating data_label
data_label = []
for i in tqdm(range(len(labels))):
    for j in get_embeddings(data[i], labels[i]):
        data_label.append(j)

# Step 6: Create DataFrame
if data_label:
    df = pd.DataFrame(data_label, columns=data_label[0].keys(), index=None)

# Step 7: Ensure 'embedding' column exists and convert embeddings to numpy arrays
if 'embedding' in df.columns:
    X = np.array([np.array(emb['embedding']) if isinstance(emb, dict) and 'embedding' in emb else np.zeros(768) for emb in df['embedding']])
    y = df['label']
else:
    raise KeyError("The 'embedding' column does not exist in the DataFrame.")

# Step 8: Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 9: KFold cross-validation
kf = KFold(n_splits=10, shuffle=True)
accs = []
f1_macros = []

for train_index, test_index in kf.split(X, y_encoded):
    classifier = AutoModelForSequenceClassification.from_pretrained(model_name)
    train_X, test_X = X[train_index], X[test_index]
    train_y, test_y = y_encoded[train_index], y_encoded[test_index]
    
    # Ensure the classifier is trained with compatible inputs
    classifier.train()  # You may need to adjust this if using a custom training loop
    classifier.train(train_X, train_y)  # This is a placeholder and will require adjustments if using PyTorch
    y_pred = classifier.predict(test_X)  # This line might need an update based on your PyTorch setup
    
    accs.append(accuracy_score(test_y, y_pred))
    f1_macros.append(f1_score(test_y, y_pred, average='macro'))

# Step 10: Save metrics
metrics = {
    'Mean Accuracy': np.mean(accs),
    'Mean F1 Macro': np.mean(f1_macros)
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Save the trained model
joblib.dump(classifier, 'xlm_roberta_model.pkl')
