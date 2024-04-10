""" Train a decision tree to perform time series classification using SAX symbols """

from aeon.transformations.collection.dictionary_based import SAX
from aeon.datasets import load_from_tsv_file
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path


def load_dataset(name, split):
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / "datasets_ucr"

    if split == "train":
        path = data_dir / f"{name}/{name}_TRAIN.tsv"
    elif split == "test":
        path = data_dir / f"{name}/{name}_TEST.tsv"
    else:
        raise Exception('Split should be either "train" or "test"')

    return load_from_tsv_file(str(path))


# Define parameters and paths
n_segments = 128
alphabet_size = 8

# Load datasets
dataset = "Strawberry"
X_train, y_train = load_dataset(dataset, "train")
X_test, y_test = load_dataset(dataset, "test")

# Get SAX symbols
sax = SAX(n_segments=n_segments, alphabet_size=alphabet_size)
X_train_sax = sax.fit_transform(X_train).squeeze()
X_test_sax = sax.fit_transform(X_test).squeeze()

# Fit decision tree classifier
clf = DecisionTreeClassifier().fit(X_train_sax, y_train)

# Calculate F1 score
y_pred = clf.predict(X_test_sax)
f1 = f1_score(y_test, y_pred, average='macro')
acc = accuracy_score(y_test, y_pred)
print(f"F1 score: {f1:.2f}")
print(f"Accuracy: {acc:.2f}")
