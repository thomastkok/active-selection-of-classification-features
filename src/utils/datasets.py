import os

import pandas as pd

from config import ROOT_DIR


def read_dataset(filename, shuffle=False):
    """Reads a given dataset.

    Args:
        filename (str): [The filename of the dataset.]
        shuffle (bool, optional): [If True, the rows of the dataset are shuffled.].
            Defaults to False.

    Returns:
        DataFrame: Pandas DataFrame of the dataset.
    """
    ds = pd.read_csv(filename)
    if shuffle:
        ds = ds.sample(frac=1).reset_index(drop=True)
    return ds


def prepare_dataset(
    dataset="Schizophrenia",
    selection_features=None,
    classification_features=None,
    label=None,
    preprocessed=True,
    shuffle_dataset=False,
):
    """Prepares the dataset to use in the experiment.

    Args:
        dataset (str, optional): The name of a known dataset, or filename of an unknown
            dataset. Defaults to "BreastCancer".
        selection_features ([str], optional): The names of the selection features, for
            an unknown dataset. Defaults to None.
        classification_features ([str], optional): The names of the classification
            features, for an unknown dataset. Defaults to None.
        label (str, optional): The name of the label column, for an unknown dataset.
            Defaults to None.
        shuffle_dataset (bool, optional): If true, shuffles the row of the dataset.
            Defaults to False.

    Raises:
        ValueError: Checks if the name of the dataset is either known or an existing
            file.
        ValueError: Checks for an unknown dataset if the splits are defined.

    Returns:
        DataFrame, DataFrame, Series: Returns the selection features, classification
            features and labels.
    """
    base_path = (
        os.path.join(ROOT_DIR, "datasets", "processed")
        if preprocessed
        else os.path.join(ROOT_DIR, "datasets", "raw")
    )
    synth_path = os.path.join(ROOT_DIR, "datasets", "synth")
    datasets = {
        "Schizophrenia": os.path.join(base_path, "schizo_dataset.csv"),
        "BreastCancer": os.path.join(base_path, "breast_cancer.csv"),
        "HeartDisease": os.path.join(base_path, "heart_disease.csv"),
        "Wine": os.path.join(base_path, "wine.csv"),
        "Iris": os.path.join(base_path, "iris.csv"),
        "Direct": os.path.join(synth_path, "direct.csv"),
        "Indirect": os.path.join(synth_path, "indirect.csv"),
        "None": os.path.join(synth_path, "none.csv"),
    }
    splits = {
        "Schizophrenia": (["scode", "age", "IQ_total"], [], "dcode"),
        "BreastCancer": (["Age", "BMI"], [], "Label"),
        "HeartDisease": (["age", "sex", "cp"], [], "target"),
        "Wine": (["alcohol", "color", "hue"], [], "name"),
        "Iris": (["sepal_length", "petal_length"], [], "species"),
    }
    synth = ["Direct", "Indirect", "None"]
    if dataset not in datasets:
        try:
            if dataset.endswith(".csv"):
                ds = read_dataset(dataset, shuffle_dataset)
            else:
                ds = read_dataset(dataset + ".csv", shuffle_dataset)
        except FileNotFoundError:
            raise ValueError(f"{dataset} is not a known dataset nor an existing file.")
        if not selection_features or not classification_features or not label:
            raise ValueError("Feature split must be defined for unknown datasets.")
    else:
        ds = read_dataset(datasets[dataset], shuffle_dataset)

    if dataset in splits:
        selection_features, classification_features, label = splits[dataset]
        if not classification_features:
            classification_features = [
                x for x in list(ds.columns) if x not in selection_features + [label]
            ]
    elif dataset in synth:
        selection_features, classification_features, label = (
            [col for col in ds if col.startswith("sf")],
            [col for col in ds if col.startswith("cf")],
            "y",
        )
    sf, cf, y = ds[selection_features], ds[classification_features], ds[label]
    return sf, cf, y
