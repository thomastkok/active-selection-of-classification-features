import logging
from functools import partial, partialmethod

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from scipy.stats import uniform
from sklearn.exceptions import NotFittedError
from sklearn.metrics import get_scorer, make_scorer
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.utils.validation import check_is_fitted


class Model:
    """Model object, which is maintained alongside the sampled Dataset object. This
    model can then be used to evaluate the current goodness of the sampled instances.

    Attributes:
        model (Estimator): The given classifier model.
        dataset (Dataset): The corresponding Dataset object.
    """

    def __init__(self, model, dataset, tune=False):
        """Initializes the Model object.

        Args:
            model (Estimator): The classifier model, which is used.
            dataset (Dataset): The Dataset object, on which the model is fitted.
        """
        self.model = model
        self.dataset = dataset
        self.retune = None
        if tune:
            self.retune = 10
            self.params = {
                "LogisticRegression": [
                    hp.choice("penalty", ["l1", "l2"]),
                    hp.lognormal("C", 0, 1),
                ],
                "SVC": {
                    "C": hp.lognormal("C", 0, 1),
                    "gamma": hp.lognormal("gamma", 0, 1),
                },
            }[self.model.__class__.__name__]
            self.trials = Trials()
            silent_loggers = ["hyperopt.tpe", "hyperopt.pyll.base", "hyperopt.fmin"]
            for logger in silent_loggers:
                logging.getLogger(logger).setLevel(logging.ERROR)

    def fit(self):
        """Fits the model, using the complete set of instances from the dataset."""
        x = self.dataset.get_complete_instances()[self.dataset.get_cf_names()]
        y = self.dataset.get_complete_instances()[self.dataset.get_y_name()]
        if y.nunique() == 1:
            return 0
        if (
            self.model.__class__.__name__ == "KNeighborsClassifier"
            and len(x.index) < self.model.n_neighbors
        ):
            return 0
        if self.retune:
            self.tune_hyperparams(x, y, metric="f1")
        self.model.fit(x, y)
        return 1

    def score(self, x, y):
        """Returns the accuracy of the model.

        Args:
            x (DataFrame): The classification features as input.
            y (Series): The labels.

        Returns:
            float: The accuracy of the model, when evaluated on x and y.
        """
        try:
            check_is_fitted(self.model, attributes=None)
        except NotFittedError:
            return 0
        return self.model.score(x, y)

    def evaluate(self, x, y, metric):
        """Evaluates the model, with given data and metric.

        Args:
            x (DataFrame): The classification features as input.
            y (Series): The labels.
            metric (str or scoring func): The metric, on which to evaluate the model.

        Returns:
            float: The metric score of the model, when evaluated on x and y.
        """
        try:
            check_is_fitted(self.model, attributes=None)
        except NotFittedError:
            return 0
        if type(metric) is str:
            metric = get_scorer(metric)._score_func
        y_pred = self.predict(x)
        return metric(y, y_pred)

    def predict(self, cf):
        """Predicts the labels for the given classification features."""
        return self.model.predict(cf)

    def tune_hyperparams(self, x, y, metric):
        """Tunes the hyperparameters of the model, using cross-validation."""
        if len(x) == 0 or len(x) % self.retune != 0:
            return
        if type(metric) is str:
            metric = get_scorer(metric)._score_func
        cv = LeaveOneOut() if len(x) == 10 else 10
        # if len(x) > self.retune and len(x) > 10:
        #     self.update_trials(len(x))
        fn = partial(self.loss_fn, x, y, metric, cv)
        best = fmin(
            fn=fn,
            space=self.params,
            algo=tpe.suggest,
            max_evals=len(self.trials) + 10,
            trials=self.trials,
            show_progressbar=False,
        )
        self.model.set_params(**best)

    def loss_fn(self, x, y, metric, cv, params):
        self.model.set_params(**params)
        return {
            "loss": 1
            - np.mean(
                cross_val_score(self.model, x, y, scoring=make_scorer(metric), cv=cv)
            ),
            "status": STATUS_OK,
        }

    def update_trials(self, len_ds):
        prev_len = len_ds - self.retune
        for res in self.trials.results:
            res["loss"] = (prev_len * res["loss"] + 0.5 * self.retune) / len_ds
