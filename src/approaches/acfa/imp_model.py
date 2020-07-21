from abc import ABC, abstractmethod


class ImputationModel(ABC):
    """Model which imputes missing classification features. Trains using the
    selection feature set as input (and optionally the label set), and the
    classification feature set as output. Not a true imputation model, as it
    reduces to an input to output problem, not a missing-value problem."""

    def __init__(self):
        pass

    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def impute(self, x):
        pass
