import pandas as pd
from numpy import mean
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from tqdm import tqdm

from src.base.sampler import Sampler


class PIUSampler(Sampler):
    """Perfect Information Utility sampler, impossible to replicate as uses the
    oracle for unknown information."""

    def sample(self):
        """Samples the next instance by querying the oracle for each instance, and with
        perfect information simulates the updated model."""
        if len(self.dataset.complete.index) < self.initial_batch_size:
            return self.initial_sample()

        best = 0, None
        instances = list(self.dataset.incomplete.index)
        for instance in instances:
            features = self.oracle.query(instance, False)

            cf = self.oracle.query(instance, True)
            y = pd.Series(
                [self.dataset.incomplete.loc[instance][self.dataset.get_y_name()]],
                index=[instance],
            )
            score = self.estimate_utility(cf, y)

            if score > best[0]:
                best = score, instance
        return best[1]

    def estimate_utility(self, cf, y):
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
        cv = LeaveOneOut() if len(y) < 10 else 5
        return mean(cross_val_score(self.clf, new_x, new_y, cv=cv))

    def cv_extended(self, x_cv, y_cv, x_test, y_test, cv=5):
        """Performs cross validation with additional test data for the scoring."""
        results = []
        kf = KFold(n_splits=cv) if len(y_cv) > 2 * cv else LeaveOneOut()
        for train_index, val_index in kf.split(x_cv):
            x_train, y_train = x_cv.iloc[train_index], y_cv.iloc[train_index]
            x_test, y_test = (
                pd.concat([x_cv.iloc[val_index], x_test]),
                pd.concat([y_cv.iloc[val_index], y_test]),
            )
            self.clf.fit(x_train, y_train)
            results.append(self.clf.score(x_test, y_test))
        return results

    def get_init(self):
        return 4

    def to_string(self):
        return "perfect-info-utility"
