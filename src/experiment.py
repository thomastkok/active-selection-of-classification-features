import pandas as pd
from numpy import mean
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler
from sklearn.svm import SVC
from tqdm import tqdm, trange

from src.base.dataset import Dataset
from src.base.model import Model
from src.base.oracle import Oracle
from src.base.sampler import Sampler
from src.baselines.random import RandomSampler
from src.utils.datasets import prepare_dataset
from src.utils.plot import plot_curve


def run_experiment(
    sf,
    cf,
    y,
    clf=SVC(),
    sampler=RandomSampler(),
    perf_measure="accuracy",
    k=5,
    n=1,
    avg_folds=True,
    to_file=None,
    plot=False,
):
    """Performs the active learning experiment.

    Args:
        sf (DataFrame): The dataset of selection features.
        cf (DataFrame): The dataset of classificiaton features.
        y (Series): The labels to predict.
        clf (Estimator, optional): The classifier model. Defaults to LinearSVC().
        smpl (Sampler, optional): The strategy for sampling. Defaults to RandomSampler().
        k (int, optional): The number of folds for the cross-validation. Defaults to 5.
        n (int, optional): The number of runs. Defaults to 5.
        avg_folds (bool, optional): Whether to average the folds within each run, and
            write the combined value or write each fold separately. Defaults to True.
        to_file (str, optional): The file to write the results to. Defaults to None.
        plot (bool, optional): Whether to plot the learning curve. Defaults to False.

    Returns:
        DataFrame: The results of the experiment, where each row is a separate run
            and each column is the number of samples.
    """
    runs = pd.DataFrame()
    if n > 1:
        print(f"Running {k}-fold cross validation {n} times.")
    iterator = trange(n) if n > 1 else range(n)
    for i in iterator:
        fold_gen = KFold(k, shuffle=True).split(sf)
        folds = [{"train": train, "test": test} for train, test in fold_gen]

        results = {0: [0] * k}

        for fold in folds:
            sf_train, cf_train, y_train = (
                sf.iloc[fold["train"]],
                cf.iloc[fold["train"]],
                y.iloc[fold["train"]],
            )
            cf_test, y_test = (
                cf.iloc[fold["test"]],
                y.iloc[fold["test"]],
            )

            empty_cf = pd.DataFrame(index=cf_train.index, columns=cf_train.columns)
            dataset = Dataset(sf_train, empty_cf, y_train)
            model = Model(clone(clf), dataset)
            oracle = Oracle(pd.concat([sf_train, cf_train, y_train], axis=1), cf_train)
            sampler.inform(dataset, clone(clf), oracle=oracle)

            for i in range(1, len(y_train) + 1):
                sample = sampler.sample()
                features = oracle.query(sample)
                dataset.update(sample, features)
                model.fit()
                score = model.evaluate(cf_test, y_test, perf_measure)

                results.setdefault(i, []).append(score)

        if avg_folds:
            results.update((x, mean(y)) for x, y in results.items())
            runs = runs.append(results, ignore_index=True)
        else:
            for i in range(k):
                fold_res = pd.Series(
                    [y[i] if len(y) > i else None for _, y in results.items()],
                    index=[x for x, _ in results.items()],
                )
                runs = runs.append(fold_res, ignore_index=True)

    if plot:
        plot_curve("Learning curve", runs)
    if to_file:
        runs.to_csv(to_file, index=False)
    return runs


def search_run(sf, cf, y, clf, sampler, k, n):
    """Runs the experiment once with the given sampler, and determines a single score
    to evaluate. This method is used for the search process of hyperparameter tuning."""
    res = run_experiment(sf, cf, y, clf, sampler, k, n, True, None, False)
    auc = [sum(res.loc[x]) / len(res.loc[x]) for x in res.index]
    return mean(auc)


def grid_search(experiment: dict, sampler: Sampler, params: dict):
    """Performs a grid search for the given sampler and its parameters. The experimental
    setup must be given as well.

    Args:
        experiment (dict): The experimental setup. The following parameters must be
            present (in order): sf, cf, y, clf, k, n.
        sampler (Sampler): The Sampler object for which we are tuning the hyperparameters.
        params (dict): The parameters for the sampler, defined as
            {'name': [option, option, option]}.

    Returns:
        DataFrame: Contains each combination of parameters, and its resulting score (AUC).
    """
    sf, cf, y, clf, k, n = experiment.values()
    param_grid = ParameterGrid(params)
    columns = list(params) + ["auc"]
    search = pd.DataFrame(index=range(len(list(param_grid))), columns=columns)
    for i, p in enumerate(tqdm(param_grid)):
        smpl = sampler(**p)
        res = search_run(sf, cf, y, clf, smpl, k, n)
        p["auc"] = res
        search.loc[i] = p
    return search


def random_search(experiment: dict, sampler: Sampler, params: dict, n_runs: int):
    """Performs a random search for the given sampler and its parameters. The experimental
    setup must be given as well.

    Args:
        experiment (dict): The experimental setup. The following parameters must be
            present (in order): sf, cf, y, clf, k, n.
        sampler (Sampler): The Sampler object for which we are tuning the hyperparameters.
        params (dict): The parameters for the sampler, defined as
            {'name': [min, max, type]}.
        n (int): The number of runs.

    Returns:
        DataFrame: Contains each combination of parameters, and its resulting score (AUC).
    """
    sf, cf, y, clf, k, n = experiment.values()
    param_grid = {
        k: randint(a, b) if c == int else uniform(a, b)
        for k, (a, b, c) in params.items()
    }
    param_sampler = ParameterSampler(param_grid, n_runs)
    columns = list(params) + ["auc"]
    search = pd.DataFrame(index=range(len(list(param_grid))), columns=columns)
    for i, p in enumerate(tqdm(param_sampler)):
        smpl = sampler(**p)
        res = search_run(sf, cf, y, clf, smpl, k, n)
        p["auc"] = res
        search.loc[i] = p
    return search
