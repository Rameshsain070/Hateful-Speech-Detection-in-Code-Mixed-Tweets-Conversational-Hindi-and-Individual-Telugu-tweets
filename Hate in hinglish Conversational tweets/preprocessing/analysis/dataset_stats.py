import pickle
from collections import Counter
import pandas as pd


def analyze_sequence_data(pkl_path):
    data = pickle.load(open(pkl_path, "rb"))

    labels = [item["label"] for item in data]
    lengths = [item["length"] for item in data]

    label_counts = Counter(labels)
    length_counts = Counter(lengths)

    summary = {
        "total_samples": len(data),
        "label_distribution": dict(label_counts),
        "sequence_length_distribution": dict(length_counts)
    }

    return summary


if __name__ == "__main__":
    stats = analyze_sequence_data("data/processed/sequence_train.pkl")
    for k, v in stats.items():
        print(f"{k}: {v}")