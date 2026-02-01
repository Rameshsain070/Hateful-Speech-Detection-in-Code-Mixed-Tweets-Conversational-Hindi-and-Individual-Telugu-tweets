import pickle
import matplotlib.pyplot as plt
from collections import Counter


def plot_label_distribution(pkl_path, save_path):
    data = pickle.load(open(pkl_path, "rb"))
    labels = [item["label"] for item in data]
    counts = Counter(labels)

    plt.figure()
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.savefig(save_path)
    plt.close()


def plot_sequence_lengths(pkl_path, save_path):
    data = pickle.load(open(pkl_path, "rb"))
    lengths = [item["length"] for item in data]
    counts = Counter(lengths)

    plt.figure()
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Sequence Length")
    plt.ylabel("Count")
    plt.title("Sequence Length Distribution")
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    plot_label_distribution(
        "data/processed/sequence_train.pkl",
        "docs/figures/class_balance.png"
    )

    plot_sequence_lengths(
        "data/processed/sequence_train.pkl",
        "docs/figures/sequence_lengths.png"
    )