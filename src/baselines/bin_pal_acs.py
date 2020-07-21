import numpy as np
import pandas as pd
from scipy.stats import beta

from src.base.sampler import Sampler


class BinaryPAL_ACS_Sampler(Sampler):
    def __init__(self, *args, **kwargs):
        self.n_p = kwargs.pop("n_p") if "n_p" in kwargs else 25
        self.M = kwargs.pop("M") if "M" in kwargs else 1
        self.step_size = kwargs.pop("step_size") if "step_size" in kwargs else 0.01
        super().__init__(*args, **kwargs)

    def inform(self, *args, **kwargs):
        super().inform(*args, **kwargs)
        self.n_features = len(self.dataset.get_cf_names())

    def sample(self):
        self.update()

        if len(self.dataset.complete.index) < self.initial_batch_size:
            return self.initial_sample()

        known_cf = self.dataset.complete[self.dataset.get_cf_names()]
        labels = [0, 1]
        class_instances = [
            self.dataset.complete[self.dataset.complete == label].index
            for label in labels
        ]

        n_known = [
            len(self.dataset.complete.index & class_instances[label])
            for label in labels
        ]
        weights = [np.mean(n_known) + 1 / (i + 1) for i in n_known]

        evaluation_instances = np.array(
            [
                self.sample_from_label(class_instances[label], self.n_p)
                for label in labels
            ]
        )
        sampled_instances = np.array(
            [self.sample_from_label(class_instances[label], self.M) for label in labels]
        )

        known_data = [
            known_cf[known_cf.index.isin(class_instances[label])] for label in labels
        ]
        current_performance = np.array(
            [
                [self.estimate_exp_perf(i, known_data) for i in instances]
                for label, instances in zip(labels, evaluation_instances)
            ]
        )

        expected_performance = np.array(
            [
                [
                    self.estimate_exp_perf(
                        i,
                        known_data,
                        sampled_data=sampled_instances[label],
                        sampled_label=list(labels).index(label),
                    )
                    for i in instances
                ]
                for label, instances in zip(labels, evaluation_instances)
            ]
        )

        diff = expected_performance - current_performance
        gain = [diff[i] * weights[i] for i in labels]
        pgain = [np.mean(gain[i]) / self.M for i in labels]

        return self.sample_class(np.argmax(pgain))

    def sample_from_label(self, label, n):
        """Samples n pseudo-instances from the distribution of the given class."""
        n_known = len(self.dataset.complete.index & label)
        ratio = n_known / (n_known + 2)

        samples = []
        for _ in range(n):
            if self.rng.random() <= ratio:
                mu = self.dataset.complete[self.dataset.get_cf_names()].loc[
                    self.rng.choice(self.dataset.complete.index & label)
                ]
                samples.append(
                    [
                        self.rng.normal(mu[i], self.parzen_window[i])
                        for i in range(self.n_features)
                    ]
                )
            else:
                samples.append(
                    [
                        self.rng.uniform(self.lower_bound[i], self.upper_bound[i])
                        for i in range(self.n_features)
                    ]
                )
        return np.array(samples)

    def estimate_exp_perf(
        self, p_instance, known_data, sampled_data=None, sampled_label=None
    ):
        """Estimates the expected performance for a given pseudo-instance and its assigned
        label, given the already known data."""
        n, k = self.get_label_stats(
            p_instance,
            known_data,
            sampled_data=sampled_data,
            sampled_label=sampled_label,
        )
        max_k = max(k)
        return beta.mean(max_k + 1, n - max_k + 1)

    def get_label_stats(
        self, p_instance, known_data, sampled_data=None, sampled_label=None
    ):
        """Determines the label statistics (n and k) for a given pseudo-instance and its
        assigned label, given the already known data."""
        n, k = 0, [0, 0]
        for i, label in enumerate(known_data):
            freq = 0
            if sampled_data is not None and sampled_label == i:
                label = label.append(
                    pd.DataFrame(
                        sampled_data,
                        columns=self.dataset.get_cf_names(),
                        index=[f"s{i}" for i in range(self.M)],
                    )
                )
            for instance in label.iterrows():
                freq += np.exp(
                    (np.linalg.norm(instance[1] - p_instance) ** 2)
                    / (2 * np.mean(self.parzen_window) ** 2)
                )
            n += freq
            k[i] = freq
        return n, k

    def update(self):
        """Updates the relevant parameters to the PAL-ACS method."""
        self.upper_bound = [
            self.dataset.complete[cf].max() for cf in self.dataset.get_cf_names()
        ]
        self.lower_bound = [
            self.dataset.complete[cf].min() for cf in self.dataset.get_cf_names()
        ]
        self.parzen_window = [
            (self.upper_bound[i] - self.lower_bound[i]) / 10
            for i in range(self.n_features)
        ]

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
        return 2

    def to_string(self):
        return "binary-pal-acs"
