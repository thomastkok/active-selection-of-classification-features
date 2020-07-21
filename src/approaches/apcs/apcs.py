from abc import abstractmethod
from itertools import product

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

from src.base.sampler import Sampler


class APCSSampler(Sampler):
    """Samples the next instance based on the Active Pseudo-Class Selection
    based approach.

    The general method for this approach is as follows:
        Before sampling
        ---------------
        Construct (n_partitions) partitions of the feature space, which combines
            the selection features and labels.

        During sampling
        ---------------
        1. For each partition, select the most informative pseudo-class using any
            known Active Class Selection method.
        2. Add one vote to each instance within this most informative pseudo-class.
        3. Return the instance with the most votes.

    Attributes:
        n_classes (int): The number of pseudo-classes created within each partition.
        n_partitions (int): The number of partitions created by bootstrapping.
        pseudo_classes (list): The list of partitions created by bootstrapping.
    """

    pseudo_classes = []

    def __init__(self, *args, **kwargs):
        self.n_classes = kwargs.pop("n_classes") if "n_classes" in kwargs else 3
        self.n_partitions = (
            kwargs.pop("n_partitions") if "n_partitions" in kwargs else 10
        )
        super().__init__(*args, **kwargs)

    def inform(self, dataset, clf, oracle):
        super().inform(dataset, clf, oracle)
        df = pd.concat([self.dataset.sf, self.dataset.y], axis=1)
        self.pseudo_classes = [
            self.construct_pseudo_classes(
                df.sample(len(df), replace=True), df, name=str(i)
            )
            for i in range(self.n_partitions)
        ]

    def sample(self):
        """Samples the next instance via active class selection, using the
        pseudo-class labels and the implemented ACS method."""
        self.update()

        if len(self.dataset.complete.index) < self.initial_batch_size:
            return self.initial_sample()

        votes = pd.Series(
            [0] * len(self.dataset.incomplete.index),
            index=self.dataset.incomplete.index,
        )
        for partition in self.pseudo_classes:
            votes[self.get_instances(self.acs(partition), partition)] += 1
        return self.rng.choice(votes[votes == votes.max()].index)

    def construct_pseudo_classes(self, bootstrap, data, name=None):
        """Constructs pseudo-classes, by performing k-means clustering followed by
        a decision tree.

        Returns:
            Series: The selected pseudo-class for each instance.
        """
        kmeans = KMeans(n_clusters=self.n_classes).fit(bootstrap)
        partition = pd.Series(
            DecisionTreeClassifier().fit(bootstrap, kmeans.labels_).predict(data),
            index=data.index,
        )
        if name:
            partition.name = name
        return partition

    def get_instances(self, lbl, partition):
        """Gets all the instances from the given pseudo-class.

        Args:
            lbl (int): The given pseudo-class.
            partition (Series): The pseudo-class label for each instance.

        Returns:
            list: The list of all instances within this pseudo-class.
        """
        instances = list(
            set(self.dataset.incomplete.index) & set(partition[partition == lbl].index)
        )
        return instances

    def get_init(self):
        return max(5, self.n_classes * self.n_partitions)

    def initial_sample(self, n=1):
        """Used for the initial sample before the proper Pseudo-Class construction
        can work. This is achieved by greedily selecting an instance that adds to
        as many pseudo-classes as possible, until each pseudo-class has at least
        one instance."""
        best, best_instance = -1, None
        for instance in self.dataset.incomplete.index:
            score = 0
            for partition in self.pseudo_classes:
                score += max(
                    0,
                    n
                    - sum(
                        [
                            partition[i] == partition[instance]
                            for i in self.dataset.complete.index
                        ]
                    ),
                )
            if score > best:
                best, best_instance = score, instance

        sampled = [best_instance] + list(self.dataset.complete.index)
        done = True
        for partition, i in product(self.pseudo_classes, range(self.n_classes)):
            if sum([partition[j] == i for j in sampled]) < n:
                done = False
                break
        if done:
            self.initial_batch_size = len(sampled)

        return best_instance

    @abstractmethod
    def acs(self, partition):
        pass

    @abstractmethod
    def update(self):
        pass

    def to_string(self):
        return "apcs"
