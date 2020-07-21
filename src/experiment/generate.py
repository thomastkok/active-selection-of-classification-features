import os
import string
import sys
from itertools import product
from random import choices, randint

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

from config import ROOT_DIR


def generate_experiment(
    seed, method, method_params, n_folds, n_repetitions, dataset, classifier,
):
    if not seed:
        raise ValueError("Seed must be given.")
    elif method not in ("ACFA", "APCS", "EM", "Baseline"):
        raise ValueError("{} is not an existing method.".format(method))
    elif not n_folds:
        raise ValueError("Number of folds must be given.")
    elif not n_repetitions:
        raise ValueError("Number of repetitions must be given.")
    elif not dataset:
        raise ValueError("Dataset must be given.")
    elif not classifier:
        raise ValueError("Classifier must be given.")
    elif method == "ACFA":
        if "utility" not in method_params:
            raise ValueError(
                "Utility method must be given for Active Classification \
            Feature Acquisition method."
            )
        if method_params["utility"] == "dual" and "lambd" not in method_params:
            raise ValueError("Lambda must be given for the Dual Objective.")
        if method_params["utility"] in ("dual", "avid") and "B" not in method_params:
            raise ValueError("Number of instances must be given for AVID.")
    elif method == "APCS":
        if "acs" not in method_params:
            raise ValueError(
                "ACS method must be given for Active Pseudo-Class \
            Selection method."
            )
        if "n_classes" not in method_params:
            raise ValueError("Number of classes must be given for APCS method.")
        if "n_partitions" not in method_params:
            raise ValueError("Number of partitions must be given for APCS method.")
    elif method == "EM":
        if "n_clusters" not in method_params:
            raise ValueError("Number of clusters must be given for EM method.")

    config = {
        "seed": seed,
        "method": method,
        "method_params": method_params,
        "n_folds": n_folds,
        "n_repetitions": n_repetitions,
        "dataset": dataset,
        "classifier": classifier,
    }
    submethod = (
        method_params["acs"]
        if method == "APCS"
        else method_params["utility"]
        if method == "ACFA"
        else method_params["ref"]
        if method == "Baseline"
        else ""
    )
    params = (
        "nclasses-{}".format(method_params["n_classes"])
        if method == "APCS"
        else "lambda-{}".format(method_params["lambd"])
        if method == "ACFA" and submethod != "Prob"
        else "b-{}".format(method_params["b"])
        if method == "ACFA" and submethod == "Prob"
        else os.path.join(
            method_params["utility"], "nclusters-{}".format(method_params["n_clusters"])
        )
        if method == "EM"
        else ""
    )
    config_path = os.path.join(
        ROOT_DIR,
        "experiments",
        "case-study",
        "no-iq" if "no-iq" in dataset else "iq",
        classifier,
        method,
        submethod,
        params,
    )
    if not os.path.exists(config_path):
        os.makedirs(config_path)
    with open(os.path.join(config_path, "config.yaml"), "w") as conf:
        yaml.dump(config, conf)


def generate_experiments(
    seed=randint(0, 10 ** 9),
    methods=[],
    params=None,
    n_folds=10,
    n_repetitions=5,
    dataset=[],
    classifier=[],
):
    for combination in product(methods, dataset, classifier):
        mth, ds, clf = combination
        c_params = params[mth] if len(methods) > 1 else params
        for params_combination in product(*c_params.values()):
            generate_experiment(
                seed=seed,
                n_folds=n_folds,
                n_repetitions=n_repetitions,
                method=mth,
                dataset=ds,
                classifier=clf,
                method_params=dict(zip(c_params.keys(), params_combination)),
            )


def run(argv):
    generate_experiments(
        seed=172663655,
        methods=["ACFA"],
        params={"utility": ["Prob"], "b": ["auto"]},
        n_folds=10,
        n_repetitions=5,
        dataset=["Schizophrenia-iq", "Schizophrenia-no-iq"],
        classifier=["LogReg", "SVM"],
    )


if __name__ == "__main__":
    run(sys.argv)
