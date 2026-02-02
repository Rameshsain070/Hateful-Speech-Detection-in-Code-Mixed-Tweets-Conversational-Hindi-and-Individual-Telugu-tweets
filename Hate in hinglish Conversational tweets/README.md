
# Hate Speech Detection in Twitter Conversations

This repository contains experimental code for hate speech detection in Twitter data, with a focus on **conversational context**.  
The project explores how different **representation strategies** and **modeling approaches** affect hate speech classification performance.

Rather than proposing a new model, the goal of this work is to **systematically compare**:
- contextual representations vs. sequence modeling
- language-model embeddings + classical classifiers vs. end-to-end transformers
- different ways of incorporating conversational structure

## 📊 Dataset

This project uses **publicly available benchmark datasets**, primarily:

- **HASOC 2021** (conversational hate speech threads)

⚠️ **Important**:
- The dataset is **not included** in this repository.
- Redistribution is restricted by the original dataset license.
- This repository assumes the data has already been downloaded and formatted locally.

## 🧠 Problem Setting

Given a Twitter conversation thread consisting of:
- a **parent tweet**
- one or more **comments**
- optional **replies**

the task is to determine whether a given tweet in the thread contains **hate speech**.

Key challenges:
* variable conversation length
* contextual dependency between tweets
* code-mixed and multilingual text

## 🧩 Preprocessing Pipeline

The preprocessing is intentionally separated from modeling and supports **three representation strategies**:

### 1️⃣ Concatenation-based Representation
All available context (parent, comment, reply) is concatenated using a separator token. this data is used by the sequential models


### 2️⃣ Mean / ABC Representation (Context Fusion)
Each tweet in a conversation is embedded separately and then fused using Mean pooling and ABC Weighting here in mean pooling parent, comment and reply are weightied equally in average, in abc weighting parent, comment and reply are weightied differenctly. The weights a, b, c here are hyperparameters. This data is used for the embedding_models

### 3️⃣ Sequence-based Representation
Tweets are represented as an **ordered sequence of embeddings**:
[parent → comment → reply]
This type of representation is used by sequential models.


## 📁 Repository Structure

```text

├── preprocessing/
│   ├── preprocess_concat.py
│   ├── preprocess_mean.py
│   ├── preprocess_sequence.py
│   └── analysis/
│       ├── dataset_stats.py
│       └── visualize_data.py
│
├── models/
│   ├── lstm_sequence.py
│   ├── embedding_classifiers.py
│   └── transformer_classifier.py
│
├── utils/
│   └── metrics.py
├──requirements.txt
└── README.md
```

## 🧪 Models and Experiments

This repository is organized around how much does conversational structure and representation choice matter for hate speech detection.

Instead of focusing on a single architecture, the experiments are designed to compare different modeling philosophies under a common preprocessing pipeline.

Broadly, the models fall into three categories.

### 1️⃣ Embedding-based Models with Classical Classifiers

In this setting, we first convert each tweet (and its conversational context) into fixed-size vector representations using pretrained language models. These embeddings are then fed into traditional machine learning classifiers.

The motivation here is practical:
- embedding-based pipelines are easier to train, faster to iterate on, and often surprisingly competitive.

MuRIL, mBERT, DistilBERT are used for embeddings:

Context fusion strategies:

Mean pooling: all available context (parent, comment, reply) is averaged

ABC weighting: parent, comment, and reply are assigned different weights

Classifiers evaluated: KNN, SVM, Random Forest


This setup allows us to isolate the effect of representation quality (which language model), fusion strategy (mean vs weighted) and classifier capacity.

All embedding-based experiments are implemented in:
```bash
models/embedding_classifiers.py
```
### 2️⃣ End-to-End Transformer Fine-tuning

In contrast to the previous setup, this category fine-tunes the entire transformer model directly for hate speech classification.

Here, the conversational context is handled through input-level concatenation, and the model learns both representation and classification jointly.
ROBERTa and IndicBERT models are used here

This setting serves as a strong baseline and helps answer the need separate fusion strategies, or does end-to-end fine-tuning already capture enough context

These experiments are implemented in:
```bash
models/transformer_classifier.py
```
### 3️⃣ Sequence-based Modeling

While concatenation and pooling ignore ordering, conversational threads are inherently sequential.
To explicitly model this structure, we treat each conversation as an ordered sequence:

 - parent → comment → reply

Each element in the sequence is represented by a pretrained embedding, and the full sequence is passed to an LSTM.

This approach focuses on temporal and contextual flow, rather than collapsing everything into a single vector.
Sequence-based experiments are implemented in:

```bash
models/lstm_sequence.py
```


📈 Evaluation Strategy

All models are evaluated using the same metrics to ensure fair comparison:
1. Accuracy
2. Precision, Recall, and F1-score (macro-averaged)
3. Confusion Matrix

Metric computation and visualization utilities are centralized in:
```bash
utils/metrics.py
```

📊 Dataset Analysis

Before training, basic dataset statistics are computed to better understand the data distribution:

1. class balance
2. sequence length distribution
3. number of conversational samples
These analyses help interpret results and avoid misleading conclusions.

Scripts for dataset inspection and visualization are available in:
```bash
preprocessing/analysis/
```

## 💾 Model Saving and Deployment Notes

All trained models are saved locally after training for downstream use (for example, in a web interface).

Examples:

saved_models/lstm_sequence.pt

saved_models/muril_svm.pkl

saved_models/roberta_transformer/

⚠️ Saved model files are intentionally not included in this repository.

If you are using the accompanying web interface update the model path inside the web application code to point to the locally saved model directory

🚀 Running the Experiments

Once preprocessing is completed, experiments can be run directly from the models directory.

```bash
# Sequence-based model
python models/lstm_sequence.py
```

```bash
# Embedding + classical classifiers
python models/embedding_classifiers.py
```

```bash
# Transformer fine-tuning
python models/transformer_classifier.py
```
Each script is self-contained and prints evaluation metrics at the end of execution.


## 📝 Notes and Limitations
- Results may vary depending on preprocessing choices, random seeds, and data splits.
- Some design decisions (such as sequence length or fusion weights) were chosen for clarity rather than optimality.
