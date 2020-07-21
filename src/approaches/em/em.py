from abc import abstractmethod
from math import ceil

import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from src.base.sampler import Sampler


class EMSampler(Sampler):
    """Samples the next instance on the Expectation Maximization-based approach.

    The general method for this approach is as follows:
        1. Use Expectation Maximization to fit a Gaussian Mixture Model on the
            complete instances.
        2. Get a flattened Gaussian Mixture Model, only including the selection
            features and labels.
        3. For each Gaussian cluster, estimate its utility.
        4. For each instance, estimate its probability of belonging to each cluster.
        5. Return the instance with the highest expected utility, by multipling each
            probability and utility.

    Attributes:
        n_clusters (int): The number of clusters for the Gaussian Mixture model.

    """

    def __init__(self, *args, **kwargs):
        self.n_clusters = kwargs.pop("n_clusters") if "n_clusters" in kwargs else 3
        super().__init__(*args, **kwargs)

    def sample(self):
        """Samples the next instance via Expectation Maximization, and the specified
        method for estimating utility of each soft cluster."""
        if len(self.dataset.complete.index) < self.initial_batch_size:
            return self.initial_sample()

        self.update()

        if self.n_clusters == "auto-dirichlet":
            gmm = BayesianGaussianMixture(
                n_components=min(20, ceil(len(self.dataset.complete) / 2)),
            ).fit(self.dataset.complete)
            n_components = len(np.unique(gmm.predict(self.dataset.complete)))
        elif self.n_clusters == "auto-bic":
            best_gmm, best_bic, best_n = None, 10 ** 9, -1
            for i in range(1, min(20, ceil(len(self.dataset.complete) / 2)) + 1):
                temp_gmm = GaussianMixture(n_components=i).fit(self.dataset.complete)
                temp_bic = temp_gmm.bic(self.dataset.complete)
                if temp_bic < best_bic:
                    best_gmm, best_bic, best_n = temp_gmm, temp_bic, i
            gmm, n_components = best_gmm, best_n
        elif self.n_clusters == "auto-aic":
            best_gmm, best_aic, best_n = None, 10 ** 9, -1
            for i in range(1, min(20, ceil(len(self.dataset.complete) / 2)) + 1):
                temp_gmm = GaussianMixture(n_components=i).fit(self.dataset.complete)
                temp_aic = temp_gmm.aic(self.dataset.complete)
                if temp_aic < best_aic:
                    best_gmm, best_aic, best_n = temp_gmm, temp_aic, i
            gmm, n_components = best_gmm, best_n
        else:
            n_components = self.n_clusters
        if self.n_clusters != "auto-bic" and self.n_clusters != "auto-aic":
            gmm = GaussianMixture(n_components=n_components).fit(self.dataset.complete)
        gmm_flattened = self.get_flattened_gmm(gmm)

        best = -(10 ** 9), None

        util = self.estimate_cluster_utility(gmm)
        for instance in self.dataset.incomplete.index:
            prob = gmm_flattened.predict_proba([self.dataset.incomplete.loc[instance]])[
                0
            ]
            score = sum([p * u for p, u in zip(prob, util)])
            if score > best[0]:
                best = score, instance

        return best[1]

    def get_flattened_gmm(self, gmm):
        """Given a Gaussian Mixture Model, returns the flattened model which only
        includes the selection features and the label, removing the classification
        features.

        Args:
            gmm (GaussianMixture): The Gaussian Mixture Model representing the entire
                data.

        Returns:
            GaussianMixture: The flattened Gaussian Mixture Model.
        """
        gmm_new = GaussianMixture(n_components=gmm.n_components)

        n_sf = len(self.dataset.get_sf_names())
        indices = list(range(n_sf)) + [-1]

        gmm_new.weights_ = gmm.weights_
        gmm_new.means_ = gmm.means_[:, indices]
        gmm_new.covariances_ = gmm.covariances_[:, indices, :][:, :, indices]
        gmm_new.precisions_ = gmm.precisions_[:, indices, :][:, :, indices]
        gmm_new.precisions_cholesky_ = gmm.precisions_cholesky_[:, indices, :][
            :, :, indices
        ]
        return gmm_new

    @abstractmethod
    def estimate_cluster_utility(self, gmm):
        pass

    @abstractmethod
    def update(self):
        pass

    def get_init(self):
        if isinstance(self.n_clusters, str):
            return 4
        return self.n_clusters if self.n_clusters > 4 else 4

    def to_string(self):
        return "em"
