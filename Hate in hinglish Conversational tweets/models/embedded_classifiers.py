# Language-model embeddings + classical classifiers for hate speech detection

import pickle
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from utils.metrics import compute_metrics, plot_confusion_matrix, print_report


# -------------------------
# Fusion strategies
# -------------------------
def fuse_representation(sample, mode="abc", weights=(0.1, 0.1, 0.3)):
    """
    mode:
      - 'mean': simple average of available embeddings
      - 'abc' : weighted fusion (paper-style)
    """
    tweet = sample["tweet_emb"]
    comment = sample["comment_emb"]
    reply = sample["reply_emb"]

    vectors = [tweet]
    if comment is not None:
        vectors.append(comment)
    if reply is not None:
        vectors.append(reply)

    if mode == "mean":
        return np.mean(vectors, axis=0)

    elif mode == "abc":
        a, b, c = weights
        vec = a * tweet
        if comment is not None:
            vec += b * comment
        if reply is not None:
            vec += c * reply
        return vec

    else:
        raise ValueError("Unknown fusion mode")


# -------------------------
# Classifier selection
# -------------------------
def get_classifier(name):
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    elif name == "svm":
        return SVC(kernel="linear", probability=True)
    elif name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        raise ValueError("Unknown classifier")


# -------------------------
# Main experiment
# -------------------------
if __name__ == "__main__":

    # Change this depending on preprocessing
    DATA_PATH = "data/processed/mean_train.pkl"
    EMBEDDING_SOURCE = "muril"   # muril | mbert | distilbert

    FUSION_MODE = "abc"          # mean | abc
    ABC_WEIGHTS = (0.1, 0.1, 0.3)

    CLASSIFIER_NAME = "svm"      # knn | svm | rf

    SAVE_PATH = f"saved_models/{EMBEDDING_SOURCE}_{CLASSIFIER_NAME}.pkl"
   
    data = pickle.load(open(DATA_PATH, "rb"))

    X, y = [], []
    for sample in data:
        X.append(fuse_representation(sample, FUSION_MODE, ABC_WEIGHTS))
        y.append(sample["label"])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = get_classifier(CLASSIFIER_NAME)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Embedding:", EMBEDDING_SOURCE)
    print("Fusion:", FUSION_MODE)
    print("Classifier:", CLASSIFIER_NAME)
    print(compute_metrics(y_test, y_pred))
    print_report(y_test, y_pred)

    plot_confusion_matrix(
        y_test, y_pred,
        f"docs/figures/{EMBEDDING_SOURCE}_{CLASSIFIER_NAME}_cm.png"
    )

    pickle.dump(clf, open(SAVE_PATH, "wb"))
    print(f"Model saved to {SAVE_PATH}")