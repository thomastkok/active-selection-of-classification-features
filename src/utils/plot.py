import pandas as pd
from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def plot_curve(title, results):
    """Plots one or more learning curves."""
    fig = plt.figure()
    for res in results.iterrows():
        lists = sorted(res[1].items())
        x, y = zip(*lists)
        sns.lineplot(x, y)
    fig.suptitle(title)
    plt.xlabel("Samples")
    plt.ylabel("Accuracy")
    plt.show()


def std_columns(df, col_names):
    """Standardizes columns of a dataframe for uniform plotting."""
    df[col_names] = preprocessing.scale(df[col_names])
    return df


def combine_experiments(df_list, std_col_names):
    """Combines multiple experiments with the same setup, but different datasets."""
    for df in df_list:
        df = std_columns(df, std_col_names)
    return pd.concat(df_list).reset_index()
