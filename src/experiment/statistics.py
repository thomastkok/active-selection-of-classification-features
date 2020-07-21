import argparse
import glob
import os
from itertools import product
from math import floor

import matplotlib.pyplot as plt
import pandas as pd

from config import ROOT_DIR

datasets = ["BreastCancer", "HeartDisease", "Wine"]
datasets = ["iq", "no-iq"]


def read_from_conf(path, metric):
    results_list = []
    for root, dirs, files in os.walk(path):
        if dirs == []:
            for fold in files:
                results = pd.read_csv(os.path.join(root, fold), index_col=0).fillna(0)
                results_list.append(results[metric])
    results_df = pd.DataFrame(results_list).T
    return results_df


def get_target(clf, ds):
    """Gets the target value for the Data Utilization Rate."""
    random_path = os.path.join(
        ROOT_DIR, "experiments", "schiz", "Baseline", "Random", clf, ds
    )
    random_data = read_from_conf(random_path, metric)
    rd_mean = random_data.mean(axis=1)
    target_score = rd_mean.mean()
    target_query = rd_mean[rd_mean >= target_score].index.min()
    return target_score, target_query


def dur(results, out=None):
    """Returns the Data Utilization Rate."""
    res = {}
    for i, j in product(["LogReg", "SVM"], datasets):
        if (i + j) not in results:
            continue
        target_score, target_query = get_target(i, j)
        results_df = results[i + j]
        rd_mean = results_df.mean(axis=1)
        res[i + "/" + j] = rd_mean[rd_mean >= target_score].index.min() / target_query
    return res


def score_at(results, at):
    """Returns the performance measure score at a specific instance in the learning curve."""
    res = {}
    for i, j in product(["LogReg", "SVM"], datasets):
        if (i + j) not in results:
            continue
        results_df = results[i + j]
        rd_mean = results_df.mean(axis=1)
        n = floor(at * len(rd_mean))
        # res[i + "/" + j] = rd_mean[n] / rd_mean[len(rd_mean) - 1]
        res[i + "/" + j] = rd_mean[n]
    return res


def auc(results):
    """Returns the Area Under the learning Curve."""
    res = {}
    for i, j in product(["LogReg", "SVM"], datasets):
        if (i + j) not in results:
            continue
        results_df = results[i + j]
        rd_mean = results_df.mean(axis=1)
        res[i + "/" + j] = sum(rd_mean) / len(rd_mean)
    return res


def summary(path, metric):
    classifiers = ["LogReg"] if "prob" in path or "Prob" in path else ["LogReg", "SVM"]

    results = {}
    for i, j in product(classifiers, datasets):
        df = read_from_conf(os.path.join(path, i, j), metric)
        results[i + j] = df

    dur_scores = dur(results)
    print("Data Utilization Rate scores:")
    print(dur_scores)

    scores_at_80 = score_at(results, 0.8)
    print("F1@80% scores:")
    print(scores_at_80)

    scores_at_90 = score_at(results, 0.9)
    print("F1@90% scores:")
    print(scores_at_90)

    scores_to_95 = instances_to(results, 0.95)
    print("95%@N scores:")
    print(scores_to_95)

    scores_to_100 = instances_to(results, 1)
    print("100%@N scores:")
    print(scores_to_100)

    auc_scores = auc(results)
    print("AUC scores:")
    print(auc_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Calculate statistics for an Active Classification Feature \
    Selection experiment."
    )
    parser.add_argument("experiment", help="The experiment's config file or folder.")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively find experiments in subfolders.",
    )
    parser.add_argument("-m", "--metric", help="Evaluation metric.")
    args = parser.parse_args()
    metric = args.metric if args.metric else "f1"

    if not args.recursive:
        summary(os.path.dirname(args.experiment), metric)
    else:
        print("Recursive evaluation not yet implemented.")
