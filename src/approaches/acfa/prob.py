import pandas as pd

from src.approaches.acfa.acfa import ACFASampler


class ProbSampler(ACFASampler):
    """Implements the Active Feature Acquisition-based Sampler by estimating utility
    with a probability based method."""

    def __init__(self, *args, **kwargs):
        self.b = kwargs.pop("b") if "b" in kwargs else "auto"
        super().__init__(*args, **kwargs)

    def estimate_utility(self, instance, cf):
        """Estimates utility by considering the probability of the predicted label being
        correctly predicted when knowing the new features."""
        if not hasattr(self.clf, "predict_proba") and not hasattr(
            self.clf, "decision_function"
        ):
            raise AttributeError(
                "Utility Estimation with prediction probability \
                                  only works with probability-based classifiers."
            )
        y = pd.Series(
            [self.dataset.incomplete.loc[instance][self.dataset.get_y_name()]],
            index=[instance],
        )
        if hasattr(self.clf, "predict_proba"):
            prob = self.clf.predict_proba([cf])[0][int(y)]
        else:
            df = self.clf.decision_function([cf])[0]
            prob = df if int(y) == 1 else 1 - df
        return self.get_score(prob)

    def get_score(self, prob):
        """Calculates the score given p and b, as in Dhurandhar2015.

        p = probability of misclassification
        b = asymmetry parameter
        score = (p(1-p))/((-2b+1)p+b**2)"""
        p = 1 - prob
        if self.b == "auto":
            b = 0.5 + 1 / (2 * len(self.dataset.complete))
        else:
            b = self.b
        if b == 0:
            return 1 - p
        elif b == 1:
            return p
        else:
            return (p * (1 - p)) / ((-2 * b + 1) * p + b * b)

    def to_string(self):
        return "acfa-prob"
