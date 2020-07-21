import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from src.approaches.apcs.apcs import APCSSampler


class RedistrictingSampler(APCSSampler):
    """Implements the Active Class Selection method of the Active Pseudo-Class
    Selection Sampler with the redistricting method.

    Attributes:
        k_folds (int): The number of folds used for updating the predicition of each
            unsampled instance.
        labels (list): The list of the predicted labels for each round.
    """

    def __init__(self, *args, **kwargs):
        self.k_folds = kwargs.pop("k_folds") if "k_folds" in kwargs else 5
        self.labels = []
        super().__init__(*args, **kwargs)

    def acs(self, partition):
        """Selects the best class to sample, using the redistricting method.
        The redistricting method works by noting which class has the most changes
        in predicted labels from the previous round.

        Args:
            partition (Series): The partition of the dataset.

        Returns:
            int: The selected pseudo-class.
        """
        redistricted = [0] * self.n_classes
        count = [0] * self.n_classes

        previous_labels = self.labels[-2][int(partition.name)]
        for index, instance in self.labels[-1][int(partition.name)].items():
            if index in previous_labels and previous_labels[index] != instance:
                redistricted[partition[index]] += 1
            count[partition[index]] += 1

        rc = [
            redistricted[i] / count[i] if count[i] > 0 else 0
            for i in range(self.n_classes)
        ]
        lbl = rc.index(max(rc))
        return lbl

    def update(self):
        """Updates the predicted labels for this round."""
        if len(self.dataset.complete.index) > 3:
            preds = []
            for partition in self.pseudo_classes:
                cv = (
                    LeaveOneOut()
                    if partition[self.dataset.complete.index].value_counts().min()
                    < self.k_folds
                    else self.k_folds
                )
                predictions = cross_val_predict(
                    self.clf,
                    self.dataset.complete[self.dataset.get_cf_names()],
                    partition[self.dataset.complete.index],
                    cv=cv,
                )
                preds.append(pd.Series(predictions, index=self.dataset.complete.index,))
            self.labels.append(preds)

    def initial_sample(self):
        return super().initial_sample(n=2)

    def to_string(self):
        return "redistricting"
