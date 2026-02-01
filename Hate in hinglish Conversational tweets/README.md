
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
