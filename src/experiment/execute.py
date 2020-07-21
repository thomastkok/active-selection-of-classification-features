import argparse
import glob
import os
import random
from math import ceil, sqrt

import numpy as np
import pandas as pd
import sklearn.metrics as skm
import telegram
from sacred import Experiment
from sacred.observers import MongoObserver, TelegramObserver
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

import src.approaches.acfa as acfa
import src.approaches.apcs as apcs
import src.approaches.em as em
import src.baselines as bl
from src.base.dataset import Dataset
from src.base.model import Model
from src.base.oracle import Oracle
from src.utils.datasets import prepare_dataset
from src.utils.metrics import mcc

ex = Experiment()


@ex.main
def run(_config):
    random.seed(_config["seed"])
    seeds = [random.randint(0, 10 ** 9) for _ in range(_config["n_repetitions"])]

    sf_sel, cf_sel, y_sel = prepare_dataset(
        _config["dataset"], phase="selection", classifier=_config["classifier"]
    )
    sf_eval, cf_eval, y_eval = prepare_dataset(
        _config["dataset"], phase="evaluation", classifier=_config["classifier"]
    )
    clf = {"SVM": SVC, "LogReg": LogisticRegression}[_config["classifier"]]
    smpl = (
        _config["method_params"]["acs"]
        if _config["method"] == "APCS"
        else _config["method_params"]["utility"]
        if _config["method"] == "ACFA"
        else "EM"
        if _config["method"] == "EM"
        else _config["method_params"]["ref"]
    )
    sampler_class = {
        "Redistricting": apcs.RedistrictingSampler,
        "AccuracyImprovement": apcs.AccImprovementSampler,
        "PAL-ACS": apcs.PAL_ACS_Sampler,
        "Dual": acfa.DualObjSampler,
        "AVID": acfa.DualObjSampler,
        "GODA": acfa.DualObjSampler,
        "Prob": acfa.ProbSampler,
        "EM": em.EMStochasticSampler,
        "Random": bl.RandomSampler,
        "PIU": bl.PIUSampler,
        "Bin-Redistricting": bl.BinaryRedistrictingSampler,
        "Bin-PAL": bl.BinaryPAL_ACS_Sampler,
    }[smpl]
    params = dict(_config["method_params"])
    if _config["method"] == "APCS":
        del params["acs"]
    elif _config["method"] == "ACFA":
        del params["utility"]
        if smpl == "AVID":
            params["lambd"] = 1
        elif smpl == "GODA":
            params["lambd"] = 0
        params["imp"] = (
            {
                "lin": acfa.LinRegImputer,
                "lasso": acfa.LassoImputer,
                "ridge": acfa.RidgeImputer,
            }[params["imp"]]
            if "imp" in params
            else acfa.LinRegImputer
        )

    measures = {
        "accuracy": skm.accuracy_score,
        "balanced_accuracy": skm.balanced_accuracy_score,
        "cohen_kappa": skm.cohen_kappa_score,
        "f1": skm.f1_score,
        "mcc": mcc,
        "precision": skm.precision_score,
        "recall": skm.recall_score,
        "roc_auc": skm.roc_auc_score,
    }

    means = dict(
        [
            (
                m,
                pd.Series(
                    [0] + [[]] * ceil(len(y_sel) - len(y_sel) / _config["n_folds"])
                ),
            )
            for m in measures
        ]
    )

    print(
        "Running {} experiments with {} folds each.".format(
            _config["n_repetitions"], _config["n_folds"]
        )
    )
    for rep, seed in enumerate(tqdm(seeds)):
        params["seed"] = seed
        sampler = sampler_class(**params)

        folds = KFold(_config["n_folds"], shuffle=True, random_state=seed).split(sf_sel)

        for j, fold in enumerate(folds):
            if _config["continue"] and os.path.exists(
                os.path.join(
                    _config["path"], "rep{}".format(rep), "fold{}.csv".format(j)
                )
            ):
                print("Skipping rep {}, fold {}.".format(rep, j))
                continue

            sf_sel_train, cf_sel_train, y_sel_train = (
                sf_sel.iloc[fold[0]],
                cf_sel.iloc[fold[0]],
                y_sel.iloc[fold[0]],
            )
            cf_eval_train, y_eval_train = (
                cf_eval.iloc[fold[0]],
                y_eval.iloc[fold[0]],
            )
            cf_eval_test, y_eval_test = (
                cf_eval.iloc[fold[1]],
                y_eval.iloc[fold[1]],
            )

            empty_cf = pd.DataFrame(
                index=cf_sel_train.index, columns=cf_sel_train.columns
            )
            dataset = Dataset(sf_sel_train, empty_cf, y_sel_train)
            model = Model(clf(random_state=seed), dataset)
            oracle = Oracle(
                pd.concat([sf_sel_train, cf_sel_train, y_sel_train], axis=1),
                cf_sel_train,
            )
            sampler.inform(dataset, clf(random_state=seed), oracle=oracle)

            results = pd.DataFrame(
                index=range(len(y_sel_train)), columns=measures.keys()
            )

            for i in range(1, len(y_sel_train) + 1):
                sample = sampler.sample()
                features = oracle.query(sample)
                dataset.update(sample, features)
                success = model.fit()
                if not success:
                    for perf_measure in measures:
                        results[perf_measure].loc[i] = 0
                        means[perf_measure].loc[i].append(0)
                else:
                    model_eval = clf().fit(
                        cf_eval_train.loc[dataset.complete.index.to_list()],
                        y_eval_train.loc[dataset.complete.index.to_list()],
                    )
                    y_pred = model_eval.predict(cf_eval_test)
                    for perf_measure in measures:
                        kwargs = (
                            {"zero_division": 0}
                            if perf_measure in ["f1", "precision", "recall"]
                            else {}
                        )
                        score = measures[perf_measure](y_eval_test, y_pred, **kwargs)
                        results[perf_measure].loc[i] = score
                        means[perf_measure].loc[i].append(score)

            if not os.path.exists(os.path.join(_config["path"], "rep{}".format(rep))):
                os.makedirs(os.path.join(_config["path"], "rep{}".format(rep)))
            results.to_csv(
                os.path.join(
                    _config["path"], "rep{}".format(rep), "fold{}.csv".format(j)
                )
            )

    for m in means.keys():
        for i, v in means[m].items():
            ex.log_scalar(m, np.mean(v), i)

    auc_score = means["accuracy"].map(np.mean).sum() / len(means["accuracy"])
    return auc_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Run an Active Classification Feature \
    Selection experiment."
    )
    parser.add_argument("config", help="The experiment configuration file or folder.")
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively find experiments in subfolders.",
    )
    parser.add_argument(
        "-t", "--telegram", action="store_true", help="Send updates to Telegram.",
    )
    parser.add_argument(
        "-m", "--mongodb", action="store_true", help="Save runs in MongoDB.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace previous runs, instead of continuing.",
    )
    args = parser.parse_args()

    if args.mongodb:
        ex.observers.append(MongoObserver())
    if args.telegram:
        ex.observers.append(TelegramObserver.from_config("telegram.json"))

    ex.add_config({"continue": not args.replace})

    if not args.recursive:
        ex.add_config(args.config)
        ex.add_config({"path": os.path.dirname(args.config)})
        ex.run()
    else:
        for filename in glob.iglob(
            os.path.join(args.config, "**/config.yaml"), recursive=True
        ):
            ex.add_config(filename)
            ex.add_config({"path": os.path.dirname(filename)})
            ex.run()
