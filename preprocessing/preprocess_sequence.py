import os
import json
import pickle
from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from clean_text import clean_text


LABEL_MAP = {
    "HOF": 1,
    "NONE": 0,
    "NOT": 0
}


class TextEmbedder:
    """
    Generates token-level or sentence-level embeddings
    suitable for sequential models.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        """
        Returns a single vector per text (mean pooled).
        """
        if not text:
            return None

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return emb.cpu().numpy()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pad_sequence(seq, max_len, emb_dim):
    padded = np.zeros((max_len, emb_dim))
    length = min(len(seq), max_len)
    padded[:length] = np.array(seq[:length])
    return padded, length


def build_sequence_samples(conversation, embedder, max_len):
    samples = []

    tweet_text = clean_text(conversation.get("tweet", ""))
    tweet_label = LABEL_MAP.get(conversation.get("label"), None)
    tweet_emb = embedder.encode(tweet_text)

    if tweet_emb is None or tweet_label is None:
        return samples

    # Tweet-only sample
    seq = [tweet_emb]
    samples.append((seq, tweet_label))

    for comment in conversation.get("comments", []):
        comment_text = clean_text(comment.get("tweet", ""))
        comment_label = LABEL_MAP.get(comment.get("label"), None)
        comment_emb = embedder.encode(comment_text)

        if comment_emb is not None and comment_label is not None:
            seq = [tweet_emb, comment_emb]
            samples.append((seq, comment_label))

        for reply in comment.get("replies", []):
            reply_text = clean_text(reply.get("tweet", ""))
            reply_label = LABEL_MAP.get(reply.get("label"), None)
            reply_emb = embedder.encode(reply_text)

            if reply_emb is not None and reply_label is not None:
                seq = [tweet_emb, comment_emb, reply_emb]
                samples.append((seq, reply_label))

    return samples


def preprocess(input_dir, output_path, model_name, max_len):
    embedder = TextEmbedder(model_name=model_name)
    processed = []

    json_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".json")
    ]

    for path in tqdm(json_files, desc="Building sequence samples"):
        conversation = load_json(path)
        samples = build_sequence_samples(conversation, embedder, max_len)

        for seq, label in samples:
            padded_seq, length = pad_sequence(
                seq,
                max_len=max_len,
                emb_dim=seq[0].shape[0]
            )
            processed.append({
                "sequence": padded_seq,
                "length": length,
                "label": label
            })

    print(f"Total sequence samples: {len(processed)}")

    with open(output_path, "wb") as f:
        pickle.dump(processed, f)


if __name__ == "__main__":
   

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_len", type=int, default=3)

    args = parser.parse_args()

    preprocess(
        args.input_dir,
        args.output,
        args.model,
        args.max_len
    )