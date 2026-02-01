# models/lstm_sequence.py
# Sequence-based hate speech detection using LSTM.

import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.metrics import compute_metrics, plot_confusion_matrix, print_report


# -------------------------
# Dataset
# -------------------------
class SequenceDataset(Dataset):
    def __init__(self, pkl_path):
        self.data = pickle.load(open(pkl_path, "rb"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq = torch.tensor(item["sequence"], dtype=torch.float32)
        label = torch.tensor(item["label"], dtype=torch.long)
        return seq, label


# -------------------------
# Model
# -------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


# -------------------------
# Train / Eval
# -------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    return y_true, y_pred


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    DATA_PATH = "data/processed/sequence_train.pkl"
    SAVE_PATH = "saved_models/lstm_sequence.pt"  

    BATCH_SIZE = 32
    EPOCHS = 10
    HIDDEN_DIM = 128
    NUM_CLASSES = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SequenceDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = dataset[0][0].shape[-1]
    model = LSTMClassifier(input_dim, HIDDEN_DIM, NUM_CLASSES).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        loss = train_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")

    y_true, y_pred = evaluate(model, loader, device)
    print(compute_metrics(y_true, y_pred))
    print_report(y_true, y_pred)

    plot_confusion_matrix(
        y_true, y_pred,
        "docs/figures/lstm_confusion_matrix.png"
    )

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")