import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from src.base.sampler import Sampler


class BinaryRedistrictingSampler(Sampler):
    """Active Binary Class Selection Redistricting based sampler."""

    def inform(self, dataset, clf, oracle):
        super().inform(dataset, clf, oracle)
        self.labels = []
        self.true_labels = self.dataset.incomplete[self.dataset.get_y_name()]

    def sample(self):
        """Samples the next instance via active class selection, using the binary
        labels and the redistricting method."""
        self.update_labels()

        if len(self.dataset.complete.index) < self.initial_batch_size:
            return self.initial_sample()

        redistricted = [0, 0]
        count = [0, 0]

        previous_labels = self.labels[-2]
        for index, instance in self.labels[-1].items():
            if index in previous_labels and previous_labels[index] != instance:
                redistricted[self.true_labels[index]] += 1
            count[self.true_labels[index]] += 1

        if redistricted[0] / count[0] == redistricted[1] / count[1]:
            return self.random_sample()
        lbl = 0 if redistricted[0] / count[0] > redistricted[1] / count[1] else 1
        return self.sample_class(lbl)

    def update_labels(self):
        """Updates the labels for the previous round."""
        if len(self.dataset.complete.index) > self.initial_batch_size - 2:
            cv = LeaveOneOut() if len(self.dataset.complete.index) < 10 else 5
            predictions = cross_val_predict(
                self.clf,
                self.dataset.complete[self.dataset.get_cf_names()],
                self.dataset.complete[self.dataset.get_y_name()],
                cv=cv,
            )
            self.labels.append(
                pd.Series(predictions, index=self.dataset.complete.index,)
            )

    def sample_class(self, lbl):
        """Sample an instance from the given (binary) class."""
        if not (lbl == 0 or lbl == 1):
            raise ValueError
        return (
            self.rng.choice(
                self.dataset.incomplete[
                    self.dataset.incomplete[self.dataset.get_y_name()] == lbl
                ].index
            )
            if len(
                self.dataset.incomplete[
                    self.dataset.incomplete[self.dataset.get_y_name()] == lbl
                ].index
            )
            > 0
            else self.rng.choice(self.dataset.incomplete.index)
        )

    def get_init(self):
        return 5

    def to_string(self):
        return "binary-acs-redistricting"
