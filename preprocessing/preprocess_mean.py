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
    Wrapper around a transformer model to generate sentence embeddings.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
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


def build_mean_samples(conversation, embedder):
    """
    Builds mean/ABC-compatible samples from one conversation.
    """

    samples = []

    tweet_text = clean_text(conversation.get("tweet", ""))
    tweet_label = LABEL_MAP.get(conversation.get("label"), None)
    tweet_emb = embedder.encode(tweet_text)

    if tweet_label is not None and tweet_emb is not None:
        samples.append({
            "tweet_emb": tweet_emb,
            "comment_emb": None,
            "reply_emb": None,
            "label": tweet_label,
            "level": "tweet"
        })

    for comment in conversation.get("comments", []):
        comment_text = clean_text(comment.get("tweet", ""))
        comment_label = LABEL_MAP.get(comment.get("label"), None)
        comment_emb = embedder.encode(comment_text)

        if comment_label is not None and comment_emb is not None:
            samples.append({
                "tweet_emb": tweet_emb,
                "comment_emb": comment_emb,
                "reply_emb": None,
                "label": comment_label,
                "level": "comment"
            })

        for reply in comment.get("replies", []):
            reply_text = clean_text(reply.get("tweet", ""))
            reply_label = LABEL_MAP.get(reply.get("label"), None)
            reply_emb = embedder.encode(reply_text)

            if reply_label is not None and reply_emb is not None:
                samples.append({
                    "tweet_emb": tweet_emb,
                    "comment_emb": comment_emb,
                    "reply_emb": reply_emb,
                    "label": reply_label,
                    "level": "reply"
                })

    return samples


def preprocess(input_dir, output_path, model_name):
    embedder = TextEmbedder(model_name=model_name)
    all_samples = []

    json_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".json")
    ]

    for path in tqdm(json_files, desc="Building mean/ABC samples"):
        conversation = load_json(path)
        samples = build_mean_samples(conversation, embedder)
        all_samples.extend(samples)

    print(f"Total mean/ABC samples: {len(all_samples)}")

    with open(output_path, "wb") as f:
        pickle.dump(all_samples, f)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")

    args = parser.parse_args()

    preprocess(args.input_dir, args.output, args.model)