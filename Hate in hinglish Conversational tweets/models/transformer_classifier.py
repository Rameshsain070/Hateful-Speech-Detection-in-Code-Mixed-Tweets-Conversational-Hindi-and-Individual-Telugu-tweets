# End-to-end transformer fine-tuning for hate speech detection

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.metrics import compute_metrics, plot_confusion_matrix, print_report


MODEL_MAP = {
    "roberta": "cardiffnlp/twitter-xlm-roberta-base",
    "indicbert": "ai4bharat/indic-bert"
}


class TextDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        df = pd.read_csv(csv_path)
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


if __name__ == "__main__":

    DATA_PATH = "data/processed/concat_train.csv"

    MODEL_NAME = "roberta"   # roberta | indicbert
    HF_MODEL = MODEL_MAP[MODEL_NAME]

    SAVE_PATH = f"saved_models/{MODEL_NAME}_transformer"

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL, num_labels=2
    )

    dataset = TextDataset(DATA_PATH, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        model.train()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed")

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(labels)
            y_pred.extend(preds)

    print("Transformer:", MODEL_NAME)
    print(compute_metrics(y_true, y_pred))
    print_report(y_true, y_pred)

    plot_confusion_matrix(
        y_true, y_pred,
        f"docs/figures/{MODEL_NAME}_transformer_cm.png"
    )

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")