from abc import ABC, abstractmethod

import numpy as np


class Sampler(ABC):
    """Sampler object which is used to select the next instance to be sampled. This is
    an abstract class and should be implemented to correspond with any possible
    approach for the Active Classification Feature Selection problem.

    Attributes:
        dataset (Dataset): The corresponding Dataset object.
        clf (Estimator): The classifier model used for instance selection.
        oracle (Oracle): The Oracle object, only should be available for
            the perfect information baseline approach.
        initial_batch_size (int): The initial batch size, for which the initial sampling
            method is used.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the Sampler object, and handles parameters that can be changed
        for experimental purposes. These are different for each implementation.

        Args:
            init (int): The initial batch size, for which the initial sampling method
                is used.
        """
        self.initial_batch_size = self.get_init()
        self.rng = (
            np.random.default_rng(kwargs.pop("seed"))
            if "seed" in kwargs
            else np.random.default_rng()
        )

    def inform(self, dataset, clf, oracle=None):
        """Adds to the Sampler object, the relevant Dataset object and the classifier
        model.

        Args:
            dataset (Dataset): The corresponding Dataset object.
            clf (Estimator): The classifier model used for instance selection.
            oracle (Oracle): The Oracle object, only should be available for
                the perfect information baseline approach.
        """
        self.dataset = dataset
        self.clf = clf
        if self.to_string() == "perfect-info-utility":
            self.oracle = oracle

    @abstractmethod
    def sample(self):
        """Called when a new instance is sampled, and must be implemented
        for a properly functioning Sampler."""
        pass

    @abstractmethod
    def to_string(self):
        """Should return a text representation of the Sampler."""
        pass

    @abstractmethod
    def get_init(self):
        """Should determine the initial sample size based on the method."""
        pass

    def initial_sample(self):
        """Samples an instance, balancing the labels yet sampled to allow an initial
        batch of samples before informed sampling can be done."""
        if len(self.dataset.complete.index) < self.initial_batch_size:
            sampled = (
                sum(self.dataset.complete[self.dataset.get_y_name()] == 0),
                sum(self.dataset.complete[self.dataset.get_y_name()] == 1),
            )
            if sampled[0] < sampled[1]:
                return self.rng.choice(
                    self.dataset.incomplete[
                        self.dataset.incomplete[self.dataset.get_y_name()] == 0
                    ].index
                )
            elif sampled[1] < sampled[0]:
                return self.rng.choice(
                    self.dataset.incomplete[
                        self.dataset.incomplete[self.dataset.get_y_name()] == 1
                    ].index
                )
            else:
                return self.random_sample()

    def random_sample(self):
        """Samples randomly one instance which has not yet been sampled."""
        return self.rng.choice(self.dataset.incomplete.index)
