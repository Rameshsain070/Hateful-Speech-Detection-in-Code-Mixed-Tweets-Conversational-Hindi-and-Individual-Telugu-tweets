import json
import os
import csv
from tqdm import tqdm

from clean_text import clean_text


SEP_TOKEN = " [SEP] " 

LABEL_MAP = {
    "HOF": 1,
    "NONE": 0,
    "NOT": 0
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_concat_samples(conversation):
    """
    Builds concatenation-based samples from a single conversation.
    Returns a list of (text, label) tuples.
    """

    samples = []

    tweet_text = clean_text(conversation.get("tweet", ""))
    tweet_label = LABEL_MAP.get(conversation.get("label"), None)

    if tweet_label is not None:
        samples.append((tweet_text, tweet_label))

    for comment in conversation.get("comments", []):
        comment_text = clean_text(comment.get("tweet", ""))
        comment_label = LABEL_MAP.get(comment.get("label"), None)

        if comment_label is not None:
            concat_text = tweet_text + SEP_TOKEN + comment_text
            samples.append((concat_text, comment_label))

        for reply in comment.get("replies", []):
            reply_text = clean_text(reply.get("tweet", ""))
            reply_label = LABEL_MAP.get(reply.get("label"), None)

            if reply_label is not None:
                concat_text = (
                    tweet_text
                    + SEP_TOKEN
                    + comment_text
                    + SEP_TOKEN
                    + reply_text
                )
                samples.append((concat_text, reply_label))

    return samples


def preprocess(input_dir, output_csv):
    all_samples = []

    json_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".json")
    ]

    for path in tqdm(json_files, desc="Processing conversations"):
        conversation = load_json(path)
        samples = build_concat_samples(conversation)
        all_samples.extend(samples)

    print(f"Total samples generated: {len(all_samples)}")

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])

        for text, label in all_samples:
            writer.writerow([text, label])


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with conversation JSON files")
    parser.add_argument("--output", required=True, help="Output CSV file")

    args = parser.parse_args()

    preprocess(args.input_dir, args.output)