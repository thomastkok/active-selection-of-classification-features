import os
from math import sin

import numpy as np
import pandas as pd

from config import ROOT_DIR
from src.base.dataset import Dataset


def get_synthetic_dataset(correlation, size, return_class=False):
    """Simplifies the dataset generation, allowing to generate a directly correlated,
    indirectly correlated, or uncorrelated synthetic dataset.

    Args:
        correlation (str): Must be either "direct", "indirect", or "none".
        size (int): The number of instances within the synthetic dataset.
        return_class (bool, optional): Whether to return the dataset as the defined
            class, returns two DataFrames and a Series otherwise. Defaults to False.

    Raises:
        ValueError: Raised when correlation is not one of the three options.

    Returns:
        DataFrame, DataFrame, Series: The selection features, classification features,
            and the labels.
        Dataset: When return_class is set to True, a Dataset object is returned.
    """
    if correlation not in ["direct", "indirect", "none"]:
        raise ValueError("Correlation must be defined properly (direct, indirect, none")

    f, g, h = (lambda x: 0, lambda x: 0, lambda x: sin(x))

    if correlation == "direct":
        f = lambda x: sin(x)
    elif correlation == "indirect":
        g = lambda x: sin(x)

    return generate_dataset(size, 2, f, g, h, 0, 1, 1, 0.2, 0, return_class)


def generate_dataset(
    n, k, f, g, h, mu_sf, var_sf, var_cf, var_y, c, return_class=False
):
    """Generates a synthetic dataset, based on the definition set out in the thesis.

    Args:
        n (int): Number of instances.
        k (int): Number of selection features, and number of classification features.
        f (func): The mapping function from the output of the selection features, to the
            mean of the classification features normal distribution.
        g (func): The mapping function from the output of the selection features,
            to the input of the label function.
        h (func): The mapping function from the output of the classification features,
            to the input of the label function.
        mu_sf (float): The mean of the distribution of the selection features.
        var_sf (float): The variance of the distribution of the selection features.
        var_cf (float): The variance of the distribution of the classification features.
        var_y (float): The noise within the label function.
        c (float): The baseline value of the label function, to determine the label.
        return_class (bool, optional): Whether to return the dataset as the defined
            class, returns two DataFrames and a Series otherwise. Defaults to False.

    Returns:
        DataFrame, DataFrame, Series: The selection features, classification features,
            and the labels.
        Dataset: When return_class is set to True, a Dataset object is returned.
    """
    sf = [[np.random.normal(mu_sf, var_sf) for j in range(k)] for i in range(n)]

    cf = [[np.random.normal(f(j), var_cf) for j in sf[i]] for i in range(n)]

    y = [
        int(
            sum([g(j) for j in sf[i]])
            + sum([h(j) for j in cf[i]])
            + np.random.normal(0, var_y)
            > c
        )
        for i in range(n)
    ]

    sf, cf, y = (
        pd.DataFrame(sf, columns=["sf_" + str(i) for i in range(k)]),
        pd.DataFrame(cf, columns=["cf_" + str(i) for i in range(k)]),
        pd.Series(y, name="y"),
    )
    if return_class:
        return Dataset(sf, cf, y)
    else:
        return sf, cf, y


if __name__ == "__main__":
    for ds in ["direct", "indirect", "none"]:
        sf, cf, y = get_synthetic_dataset(ds, 200)
        dataset = pd.concat([sf, cf, y], axis=1)
        dataset.to_csv(
            os.path.join(ROOT_DIR, "datasets", "synth", f"{ds}.csv"), index=False
        )
