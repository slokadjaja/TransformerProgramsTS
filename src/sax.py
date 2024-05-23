""" Train a decision tree to perform time series classification using SAX symbols """

import numpy as np
import matplotlib.pyplot as plt
from aeon.transformations.collection.dictionary_based import SAX
from aeon.datasets import load_from_tsv_file
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
import seaborn as sns


def plot_dataset(name, split):
    """
    Plots time series in the dataset, separated per class
    :param name: Name of the dataset
    :param split: "train" or "test"
    :return: fig, ax
    """
    path = get_dataset_path(name, split)
    X, y = load_from_tsv_file(path)
    X = np.squeeze(X)
    classes = np.unique(y)

    fig, ax = plt.subplots(len(classes), 1)

    for idx, cls in enumerate(classes):
        ax_cls = sns.lineplot(X[y == cls, :].T, ax=ax[idx], dashes=False, legend=False)
        ax_cls.set_title(f"class {cls}")

    fig.tight_layout()
    plt.show()

    return fig, ax


def get_dataset_path(name, split):
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent / "datasets_ucr"

    if split == "train":
        path = data_dir / f"{name}/{name}_TRAIN.tsv"
    elif split == "test":
        path = data_dir / f"{name}/{name}_TEST.tsv"
    else:
        raise Exception('Split should be either "train" or "test"')

    return str(path)


def main():
    # Define parameters and paths
    n_segments = 128
    alphabet_size = 8

    # Load datasets
    dataset = "Strawberry"
    X_train, y_train = load_from_tsv_file(get_dataset_path(dataset, "train"))
    X_test, y_test = load_from_tsv_file(get_dataset_path(dataset, "test"))

    # Get SAX symbols
    sax = SAX(n_segments=n_segments, alphabet_size=alphabet_size)
    X_train_sax = sax.fit_transform(X_train).squeeze()
    X_test_sax = sax.fit_transform(X_test).squeeze()

    # Fit decision tree classifier
    clf = DecisionTreeClassifier().fit(X_train_sax, y_train)

    # Calculate F1 and accuracy score
    y_pred = clf.predict(X_test_sax)
    f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    print(f"F1 score: {f1:.2f}")
    print(f"Accuracy: {acc:.2f}")


if __name__ == "__main__":
    main()