from src.approaches.acfa.imp_model import ImputationModel
from sklearn.linear_model import LinearRegression


class LinRegImputer(ImputationModel):
    """Implements the ImputationModel with a simple Linear regression model. It trains
    on the selection features and labels as input, and predicts the classification
    features as output.

    Attributes:
        model (Estimator): The linear regression model.
    """

    def __init__(self):
        self.model = LinearRegression()

    def train(self, x, y):
        self.model.fit(x, y)

    def impute(self, x):
        return self.model.predict(x)
