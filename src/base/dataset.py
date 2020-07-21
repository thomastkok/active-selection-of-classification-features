import pandas as pd


class Dataset:
    """Dataset object, which maintains the sets of features and labels, as well as
    the unsampled and sampled set of instances.

    Attributes:
        sf (DataFrame): The selection features.
        cf (DataFrame): The known classification features.
        y (Series): The labels.
        incomplete (DataFrame): The unsampled instances.
        complete (DataFrame): The sampled instances.
    """

    def __init__(self, sf, cf, y):
        """Initializes a Dataset object.

        Args:
            sf (DataFrame): The selection features.
            cf (DataFrame): The classification features, should be an empty DataFrame
                with available column names.
            y (Series): The labels.
        """
        self.sf = sf
        self.cf = cf
        self.y = y
        self.incomplete = pd.concat([sf, y], axis=1)
        self.complete = pd.DataFrame(
            columns=list(sf.columns) + list(cf.columns) + [y.name]
        )

    def update(self, index, features):
        """Updates the dataset, by adding the new classification features.

        Args:
            index (int): The index of the instance with new features.
            features (Series): The new features to add.
        """
        self.cf.loc[index] = features
        self.complete.loc[index] = features
        self.complete.loc[index][self.sf.columns] = self.sf.loc[index]
        self.complete.loc[index][self.y.name] = self.y[index]
        self.incomplete.drop(index, inplace=True)

    def get_complete_instances(self):
        return self.complete

    def get_incomplete_instances(self):
        return self.incomplete

    def get_sf_names(self):
        return self.sf.columns

    def get_cf_names(self):
        return self.cf.columns

    def get_y_name(self):
        return self.y.name
