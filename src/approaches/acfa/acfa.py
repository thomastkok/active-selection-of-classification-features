from abc import abstractmethod

import pandas as pd

from src.approaches.acfa.lin_reg import LinRegImputer
from src.base.sampler import Sampler


class ACFASampler(Sampler):
    """Samples the next instance based on the Active Feature Acquisition-based approach.

    The general method for this approach is as follows:
        1. Estimate the missing classification features for each instance.
        2. Estimate the utility for each instance, given the estimated
            classification features.
        3. Return the instance with the highest expected utility.

    Attributes:
        imp_model (ImputationModel): The imputation model used for estimating the
            classification features.
    """

    def __init__(self, *args, **kwargs):
        self.imp = kwargs.pop("imp") if "imp" in kwargs else LinRegImputer
        self.imp_model = self.imp()
        super().__init__(*args, **kwargs)

    def sample(self):
        """Samples an unsampled instance, by estimating the classification features,
        and determining the instance with the highest utility."""
        if len(self.dataset.complete.index) < self.initial_batch_size:
            return self.initial_sample()

        self.update()

        best = -(10 ** 9), None
        for instance in self.dataset.incomplete.index:
            predicted_cf = self.imp_model.impute(
                [
                    pd.concat(
                        [
                            self.dataset.incomplete.loc[instance][
                                self.dataset.get_sf_names()
                            ],
                            pd.Series(
                                self.dataset.incomplete.loc[instance][
                                    self.dataset.get_y_name()
                                ],
                                index=[self.dataset.get_y_name()],
                            ),
                        ],
                    )
                ]
            )[0]
            utility = self.estimate_utility(
                instance, pd.Series(predicted_cf, index=self.dataset.get_cf_names()),
            )
            if utility > best[0]:
                best = utility, instance

        return best[1]

    @abstractmethod
    def estimate_utility(self, cf, y):
        pass

    def update(self):
        """Retrains the imputation model and the classifier."""
        self.imp_model.train(
            pd.concat(
                [
                    self.dataset.complete[self.dataset.get_sf_names()],
                    self.dataset.complete[self.dataset.get_y_name()],
                ],
                axis=1,
            ),
            self.dataset.complete[self.dataset.get_cf_names()],
        )
        self.clf.fit(
            self.dataset.complete[self.dataset.get_cf_names()],
            self.dataset.complete[self.dataset.get_y_name()],
        )

    def get_init(self):
        return 4

    def to_string(self):
        return "acfa"
