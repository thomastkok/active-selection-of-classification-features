import pandas as pd
from numpy import mean, var
from sklearn.metrics import hinge_loss, log_loss, make_scorer
from sklearn.model_selection import LeaveOneOut, cross_val_predict, cross_val_score
from sklearn.utils import resample

from src.approaches.acfa.acfa import ACFASampler
from src.approaches.acfa.lin_reg import LinRegImputer


class DualObjSampler(ACFASampler):
    """Implements the Active Feature Acquisition-based Sampler with a dual objective,
    as seen in the papers by Zheng in 2002 and 2006. It combines the variance in
    prediction with the expected usefulness of the predicted features.

    Attributes:
        lambd (float): The ratio of the dual objective. If 0, entirely based on the
            GODA function. If 1, entirely based on the AIVD function.
        imp_models (list[Estimators]): The imp_models used for estimating the variance
            in prediction.
        features (DataFrame): The features used to estimate the classification features,
            with the imputation models.
    """

    def __init__(self, *args, **kwargs):
        self.lambd = kwargs.pop("lambd") if "lambd" in kwargs else 0.5
        self.B = kwargs.pop("B") if "B" in kwargs else 10
        super().__init__(*args, **kwargs)

    def inform(self, dataset, clf, oracle):
        super().inform(dataset, clf, oracle)
        self.imp_models = [self.imp() for i in range(self.B)]
        self.features = pd.concat([self.dataset.sf, self.dataset.y], axis=1)

    def estimate_utility(self, instance, cf):
        """Estimates utility by combining the dual objective functions."""
        y = pd.Series(
            [self.dataset.incomplete.loc[instance][self.dataset.get_y_name()]],
            index=[instance],
        )
        if self.lambd == 0:
            return self.goda(cf, y)
        elif self.lambd == 1:
            return self.avid(self.features.loc[instance])
        else:
            return self.lambd * self.avid(self.features.loc[instance]) + (
                1 - self.lambd
            ) * self.goda(cf, y)

    def avid(self, feats):
        """Estimates utility by considering the variance of the prediction of B
        bootstrapped imputation models."""
        return var([x.impute([feats]) for x in self.imp_models])

    def goda(self, cf, y):
        """Estimates utility by considering the accuracy of the model after adding
        instance."""
        new_x, new_y = (
            self.dataset.complete[self.dataset.get_cf_names()].append(
                cf, ignore_index=True
            ),
            self.dataset.complete[self.dataset.get_y_name()].append(
                y, ignore_index=True
            ),
        )
        cv = (
            LeaveOneOut()
            if (len(new_y[new_y == 0]) < 5 or len(new_y[new_y == 1]) < 5)
            else 5
        )
        if self.clf.__class__.__name__ == "LogisticRegression":
            y_pred = cross_val_predict(
                self.clf, new_x, new_y, cv=cv, method="predict_proba"
            )
            return 1 - log_loss(new_y, y_pred)
        elif self.clf.__class__.__name__ == "SVC":
            y_pred = cross_val_predict(
                self.clf, new_x, new_y, cv=cv, method="decision_function"
            )
            return 1 - hinge_loss(new_y, y_pred)
        else:
            raise ValueError(f"{self.clf.__class__.__name__} not implemented.")

    def update(self):
        """Updates the imputation models with the new data."""
        super().update()
        imp_x = pd.concat(
            [
                self.dataset.complete[self.dataset.get_sf_names()],
                self.dataset.complete[self.dataset.get_y_name()],
            ],
            axis=1,
        )
        imp_y = self.dataset.complete[self.dataset.get_cf_names()]
        [imp_model.train(*resample(imp_x, imp_y)) for imp_model in self.imp_models]

    def to_string(self):
        return "acfa-dualobj"
