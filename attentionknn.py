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
import torch
import torch.nn as nn
import torch.nn.functional as F

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

print(data[0])
labels = []
for i in directories:
    try:
        with open(i + 'labels.json', encoding='utf-8') as f:
            labels.append(json.load(f))
    except FileNotFoundError:
        continue

print(labels[0])

# Step 3: Loading the SentenceTransformer model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Step 4: Define attention-based embedding function
class AttentionMechanism(nn.Module):
    def __init__(self, input_dim):
        super(AttentionMechanism, self).__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, embeddings):
        scores = self.attention(embeddings)  # Compute attention scores
        weights = F.softmax(scores, dim=0)  # Normalize scores to weights
        weighted_embeddings = torch.sum(weights * embeddings, dim=0)  # Weighted sum
        return weighted_embeddings

attention_layer = AttentionMechanism(input_dim=384)  # Input dimension of MiniLM embeddings

# Step 5: Embedding function with attention

def attention_based_ST(d, l):
    embeddings = []
    tweet_embedding = torch.tensor(model.encode(d['tweet']), dtype=torch.float32)
    tweet_data = [{'embedding': tweet_embedding, 'tweet_id': d['tweet_id'], 'label': l[d['tweet_id']]}]

    comment_embeddings = []
    for i in d['comments']:
        comment_embedding = torch.tensor(model.encode(i['tweet']), dtype=torch.float32)
        comment_embeddings.append(comment_embedding)

        if 'replies' in i.keys():
            reply_embeddings = []
            for j in i['replies']:
                reply_embedding = torch.tensor(model.encode(j['tweet']), dtype=torch.float32)
                reply_embeddings.append(reply_embedding)

            reply_embeddings = torch.stack(reply_embeddings)
            combined_embedding = attention_layer(reply_embeddings)
            comment_embeddings.append(combined_embedding)

    if comment_embeddings:
        comment_embeddings = torch.stack(comment_embeddings)
        context_embedding = attention_layer(comment_embeddings)
    else:
        context_embedding = tweet_embedding

    combined_embedding = attention_layer(torch.stack([tweet_embedding, context_embedding]))

    embeddings.append({
        'tweet_id': d['tweet_id'],
        'embedding': combined_embedding.detach().numpy(),
        'label': l[d['tweet_id']]
    })

    return embeddings

# Step 6: Populating data_label and handling empty cases
data_label = []

for i in tqdm(range(len(labels))):
    for j in attention_based_ST(data[i], labels[i]):
        data_label.append(j)

# Step 7: Check data_label is not empty before creating the DataFrame
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

# Step 8: Convert embeddings to numpy arrays for model training
X = [np.array(j) for j in df.embedding]
X = np.array(X)
y = df.label

# Step 9: KFold cross-validation
kf = KFold(n_splits=10, shuffle=True)
accs = []
f1_macros = []

for train_index, test_index in kf.split(X, y):
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X[train_index], y[train_index])
    y_pred = classifier.predict(X[test_index])
    accs.append(float(accuracy_score(y[test_index], y_pred)))
    f1_macros.append(float(f1_score(y[test_index], y_pred, average='macro')))

# Step 10: Training on full dataset
d = {
    'Mean Accuracy': np.mean(np.array(accs)).item(),
    'Mean F1_macro': np.mean(np.array(f1_macros)).item(),
}

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X[:5740], y[:5740])
y_pred = classifier.predict(X[5740:])
d['Test Accuracy'] = float(accuracy_score(y[5740:], y_pred))
d['Test Macro F1'] = float(f1_score(y[5740:], y_pred, average='macro'))

# Step 11: Save performance metrics as JSON
d = json.dumps(d, indent=4)
print(d)

# Step 12: Save the KNN model
joblib.dump(classifier, 'knn_model_with_attention.pkl')
