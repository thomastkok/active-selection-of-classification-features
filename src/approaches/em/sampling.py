import pandas as pd
from numpy import mean, var
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.utils import resample

from src.approaches.acfa.lin_reg import LinRegImputer
from src.approaches.em.em import EMSampler


class EMStochasticSampler(EMSampler):
    """Implements the estimation of the cluster utility of the EM Sampler, by sampling
    each component and estimating the utility for each sampled instance. These are
    then averaged to created an ultimate utility value."""

    def __init__(self, *args, **kwargs):
        self.n_samples = kwargs.pop("n_samples") if "n_samples" in kwargs else 25
        self.B = kwargs.pop("B") if "B" in kwargs else 10
        self.utility = kwargs.pop("utility") if "utility" in kwargs else "avid"
        super().__init__(*args, **kwargs)

    def inform(self, dataset, clf, oracle):
        super().inform(dataset, clf, oracle)
        self.imp_models = [LinRegImputer() for i in range(self.B)]
        self.features = pd.concat([self.dataset.sf, self.dataset.y], axis=1)

    def estimate_cluster_utility(self, gmm):
        """The utility of a cluster is determined by sampling it repeatedly,
        estimating the utility for each sample and averaging the results.

        Args:
            gmm (GaussianMixture): The flattened Gaussian Mixture Model.

        Returns:
            list: The estimated utility for each cluster.
        """
        results = []
        for c in range(gmm.n_components):
            samples = self.sample_component(gmm, c, self.n_samples)
            results.append(
                mean(
                    [
                        self.estimate_instance_utility(
                            x[1][self.dataset.get_sf_names()],
                            x[1][self.dataset.get_cf_names()],
                            x[1][self.dataset.get_y_name() :],
                        )
                        for x in samples.iterrows()
                    ]
                )
            )
        return results

    def sample_component(self, gmm, c, n_samples):
        """Samples one specific cluster.

        Args:
            gmm (GaussianMixture): The Gaussian Mixture Model to sample.
            c (int): The cluster within the model to sample.
            n_samples (int): The number of instance to sample and return.

        Returns:
            DataFrame: The sampled instances.
        """
        mean, cov = (gmm.means_[c], gmm.covariances_[c])
        df = pd.DataFrame(
            self.rng.multivariate_normal(mean, cov, size=n_samples),
            columns=self.dataset.complete.columns,
        )
        df[self.dataset.get_y_name()] = pd.Series(
            [0 if x < 0.5 else 1 for x in df[self.dataset.get_y_name()]], index=df.index
        )
        return df

    def estimate_instance_utility(self, sf, cf, y):
        """Estimates the utility for one sampled instance, depending
        on the given utility method."""
        if self.utility == "avid":
            return self.avid(sf.append(y))
        elif self.utility == "goda":
            return self.goda(cf, y)
        elif self.utility == "prob":
            return self.prob(cf, y)

    def goda(self, cf, y):
        """Estimates the utility for one sampled instance, by retraining the model
        with the instance potentially added.

        Args:
            cf (Series): The classification features of the instance.
            y (int): The label of the instance.

        Returns:
            float: The accuracy of the retrained model, which is the utility.
        """
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
        return mean(cross_val_score(self.clf, new_x, new_y, cv=cv))

    def avid(self, feats):
        """Estimates the utility for one sampled instance, by the AVID method."""
        return var([x.impute([feats]) for x in self.imp_models])

    def prob(self, cf, y):
        """Estimates the utility for one sampled instance, with the
        Probability-based score function."""
        prob = self.clf.predict_proba([cf])[0][int(y)]
        p = 1 - prob
        b = 0.5 + 1 / (2 * len(self.dataset.complete))
        return (p * (1 - p)) / ((-2 * b + 1) * p + b * b)

    def update(self):
        """Updates the imputation models with the new data."""
        imp_x = pd.concat(
            [
                self.dataset.complete[self.dataset.get_sf_names()],
                self.dataset.complete[self.dataset.get_y_name()],
            ],
            axis=1,
        )
        imp_y = self.dataset.complete[self.dataset.get_cf_names()]
        [imp_model.train(*resample(imp_x, imp_y)) for imp_model in self.imp_models]
        self.clf.fit(
            self.dataset.complete[self.dataset.get_cf_names()],
            self.dataset.complete[self.dataset.get_y_name()],
        )

    def to_string(self):
        return "em-sampling"
