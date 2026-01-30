import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import itertools

# Loading the CSV files and select only the required columns
train_df = pd.read_csv('train.csv', usecols=['Comments', 'Label'])
test_df = pd.read_csv('test.csv', usecols=['Comments', 'Label'])

# Encoding labels using LabelEncoder
label_encoder = LabelEncoder()
train_df['Label'] = label_encoder.fit_transform(train_df['Label'].astype(str))
test_df['Label'] = label_encoder.transform(test_df['Label'].astype(str))

# Loading MuRIL tokenizer and model from Hugging Face Transformers
model_name = 'google/muril-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

# Function to generate embeddings
def generate_embeddings(texts, tokenizer, model, batch_size=16):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            tokens = tokenizer(
                batch_texts, 
                truncation=True, 
                padding=True, 
                return_tensors="pt", 
                max_length=512
            )
            outputs = model(**tokens.to(model.device))
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(embeddings)

# Generating embeddings for the train and test data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model.to(device)
train_embeddings = generate_embeddings(train_df['Comments'].astype(str).tolist(), tokenizer, base_model)
test_embeddings = generate_embeddings(test_df['Comments'].astype(str).tolist(), tokenizer, base_model)

# Defining the LSTM-based classifier
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out

# Defining hyperparameter search space
hyperparameter_grid = {
    'hidden_dim': [64, 128, 256],
    'num_layers': [1, 2],
    'dropout': [0.2, 0.5],
    'learning_rate': [1e-3, 5e-4]
}

# Performing grid search
best_score = 0
best_params = None
input_dim = train_embeddings.shape[1]
output_dim = len(np.unique(train_df['Label']))
train_dataset = list(zip(torch.tensor(train_embeddings, dtype=torch.float32), torch.tensor(train_df['Label'], dtype=torch.long)))
test_dataset = list(zip(torch.tensor(test_embeddings, dtype=torch.float32), torch.tensor(test_df['Label'], dtype=torch.long)))

for params in itertools.product(*hyperparameter_grid.values()):
    params = dict(zip(hyperparameter_grid.keys(), params))
    model = LSTMClassifier(input_dim, params['hidden_dim'], output_dim, params['num_layers'], params['dropout']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Training loop
    for epoch in range(5):  
        model.train()
        for X_batch, y_batch in DataLoader(train_dataset, batch_size=16, shuffle=True):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch.unsqueeze(1))  
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    preds = []
    true_labels = []
    with torch.no_grad():
        for X_batch, y_batch in DataLoader(test_dataset, batch_size=64):
            X_batch = X_batch.to(device)
            outputs = model(X_batch.unsqueeze(1))
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            true_labels.extend(y_batch.cpu().numpy())
    
    # Computing scores
    accuracy = accuracy_score(true_labels, preds)
    if accuracy > best_score:
        best_score = accuracy
        best_params = params


model = LSTMClassifier(input_dim, best_params['hidden_dim'], output_dim, best_params['num_layers'], best_params['dropout']).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])

for epoch in range(5):  
    model.train()
    for X_batch, y_batch in DataLoader(train_dataset, batch_size=16, shuffle=True):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch.unsqueeze(1))
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Final evaluation
model.eval()
preds = []
true_labels = []
with torch.no_grad():
    for X_batch, y_batch in DataLoader(test_dataset, batch_size=64):
        X_batch = X_batch.to(device)
        outputs = model(X_batch.unsqueeze(1))
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        true_labels.extend(y_batch.cpu().numpy())

# Printing evaluation metrics
print("Best Hyperparameters:", best_params)
print("Precision:", precision_score(true_labels, preds, average='weighted'))
print("Recall:", recall_score(true_labels, preds, average='weighted'))
print("Accuracy:", accuracy_score(true_labels, preds))
print("F1 Score:", f1_score(true_labels, preds, average='weighted'))
