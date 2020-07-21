import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from src.approaches.apcs.apcs import APCSSampler


class AccImprovementSampler(APCSSampler):

    prev_acc = []
    curr_acc = []

    def __init__(self, *args, **kwargs):
        self.k_folds = kwargs.pop("k_folds") if "k_folds" in kwargs else 5
        super().__init__(*args, **kwargs)

    def acs(self, partition):
        """Selects the best class to sample, using the accuracy improvement method.
        The accuracy improvement method works by noting which class has the most increase
        in accuracy from the previous round.

        Args:
            partition (Series): The partition of the dataset.

        Returns:
            int: The selected pseudo-class.
        """

        improvement = [
            max(
                0,
                self.curr_acc[int(partition.name)][i]
                - self.prev_acc[int(partition.name)][i],
            )
            for i in range(self.n_classes)
        ]

        lbl = improvement.index(max(improvement))
        return lbl

    def update(self):
        """Updates the predicted labels, for this round."""
        if len(self.dataset.complete.index) > 3:
            self.prev_acc = self.curr_acc
            acc = []
            for partition in self.pseudo_classes:
                cv = (
                    LeaveOneOut()
                    if partition[self.dataset.complete.index].value_counts().min()
                    < self.k_folds
                    else self.k_folds
                )
                predictions = pd.Series(
                    cross_val_predict(
                        self.clf,
                        self.dataset.complete[self.dataset.get_cf_names()],
                        partition[self.dataset.complete.index],
                        cv=cv,
                    ),
                    index=self.dataset.complete.index,
                )
                scores = []
                for p_class in sorted(partition.unique()):
                    score = len(
                        predictions[predictions == p_class]
                        & partition[partition == p_class]
                    ) / len(partition[partition == p_class])
                    scores.append(score)
                acc.append(pd.Series(scores, index=sorted(partition.unique())))
            self.curr_acc = acc

    def initial_sample(self):
        return super().initial_sample(n=2)

    def to_string(self):
        return "accuracy-improvement"
