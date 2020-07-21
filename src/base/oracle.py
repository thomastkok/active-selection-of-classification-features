class Oracle:
    """Oracle object, which can be used to sample new instances.

    Attributes:
        dataset (DataFrame): The entire dataset.
        cf (DataFrame): The classification features, of all instances.
    """

    def __init__(self, dataset, cf):
        """Initializes the Oracle object.

        Args:
            dataset (DataFrame): The entire dataset, for queries for an entire instance.
            cf (DataFrame): The classification features of all instance, for
                queries for all classification features for an instance.
        """
        self.dataset = dataset
        self.cf = cf

    def query(self, id, cf_only=True):
        """Queries the features for a given instance.

        Args:
            id (int): The index of the instance queried.
            cf_only (bool, optional): Whether or not all features must be queried. If
                True, only the classification features are returned, otherwise the
                selection features and labels are also returned. Defaults to True.

        Returns:
            Series: The returned features.
        """
        if cf_only:
            return self.cf.loc[id]
        else:
            return self.dataset.loc[id]
