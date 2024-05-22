"""Unsupervised outlier detection model.

This module implements supervised outlier detection models based on
`this paper <https://www.sciencedirect.com/science/article/pii/S0167923624000290>`_.

Examples:
    Fitting the model on a dataset::

        import mlconcepts

        model = mlconcepts.UODModel()
        model.fit("dataset.csv")

    The method :meth:`mlconcepts.UODModel.fit` accepts all the arguments that
    :func:`mlconcepts.data.load` accepts; indeed, it relays to `load` the
    responsibility to load a dataset. Please see :mod:`mlconcepts.data` to see
    what formats are supported and how to add support for new formats.

    The following example shows how to generate different train-test splits
    using `sklearn`. This example assumes that the dataset 
    `mammography.mat <https://odds.cs.stonybrook.edu/mammography-dataset/>`_ is
    in the working directory.
    ::

        import mlconcepts
        import mlconcepts.data
        import sklearn.metrics
        import sklearn.model_selection

        data = mlconcepts.data.load("mammography.mat")

        skf = sklearn.model_selection.StratifiedKFold(n_splits = 4)
        for train, test in data.split(skf):
            model = mlconcepts.UODModel(
                n = 32, # quantize numerical features in 32 uniform bins
                doubletons = True # generate all feature pairs
            )
            model.fit(train)
            predictions = model.predict(test)
            print("AUC: " + str(sklearn.metrics.roc_auc_score(test.y, predictions)))


Todo:
    Add python methods to access explanation data.
"""

import mlconcepts.mlconceptscore
import mlconcepts.data
import numpy as np

from .ExplanationData import ExplanationData

class UODModel:
    """Represents an unsupervised model using FCA for outlier detection.

    This interface is a facade for `mlconceptscore`. Its usage is similar to
    the standard usage of `sklearn` models.
    """

    def __init__(self, n = 32, quantizer = "uniform", explorer = "none", 
                 singletons = True, doubletons = True, full = True):
        """Constructs a supervised outlier detection model.

        Args:
            n (int): The default number of bins for uniform quantization. 
                Defaults to 32.
            quantizer (str): The quantizer for real features. The available
                options are:

                * 'uniform' for a quantizer that breaks the range of the
                  feature into uniformly sized bins.

                Defaults to 'uniform'.
            explorer (str): Specifies the feature set exploration strategy.
                The available options are:

                * 'none' for no exploration.

                Defaults to 'none'.
            singletons (bool): Whether singleton agendas should be generated.
                Defaults to True.
            doubletons (bool): Whether doubleton agendas should be generated.
                Default to True.
            full (bool): Whether the full agenda should be generated. 
                Defaults to True.
        """
        self.model = None
        if quantizer == "uniform":
            if explorer == "none":
                self.model = mlconcepts.mlconceptscore.UODUniform(
                           n=n, singletons=singletons, doubletons=doubletons,
                           full=full
                )
            else:
                raise ValueError("The only available value for the explorer is 'none'")
        else:
            raise ValueError("The only available value for the quantizer is 'uniform'")

    def fit(self, dataset, categorical=[], labels=None, Xc=None, y=None, 
        settings={}):
        """Trains the model on a given dataset.

        Args:
            dataset: The dataset to train the model on. It could be represented in
                various formats. The type/format of the dataset is automatically
                detected and a data-loader is used accordingly.
            categorical: A list of features suggested to be categorical. Data
                loaders should automatically detect categorical features. This
                parameter is used for those categorical features which are hard
                to distinguish from numerical ones, e.g., columns containing only
                0 or 1.
            labels: The name of the labels column in the dataset.
            Xc: A dataframe containing categorical data. Some data-loaders may
                require categorical data to be separated from the numerical one.
                Specify categorical data here according to the specifications in
                the data loader.
            y: A dataframe containing labels data. Some data-loaders may require
                labels data to be separated from the rest. Specify labels data
                here according to the specifications in the data loader.
            settings: A dictionary containing custom parameters that can vary
                between different data loaders.
        """
        data = mlconcepts.data.load(dataset, categorical=categorical, 
                                    labels=labels, Xc=Xc, y=y, settings=settings)
        self.model.fit(data.X if data.X is not None else 
                    np.empty(shape=(0, 0), dtype=np.float64, order="F"),
                    data.Xc if data.Xc is not None else 
                    np.empty(shape=(0, 0), dtype=np.int32, order="F"))

    def predict(self, dataset, categorical=[], labels=None, Xc=None, y=None,
            settings={}):
        """Predicts labels for the given dataset.

        Args:
            dataset: A dataset represented in some format. The type/format of the
                dataset is automatically detected and a data-loader is used
                accordingly.
            categorical: A list of features suggested to be categorical. Data
                loaders should automatically detect categorical features. This
                parameter is used for those categorical features which are hard to
                distinguish from numerical ones, e.g., columns containing only 0 or
                1.
            labels: The name of the labels column in the dataset.
            Xc: A dataframe containing categorical data. Some data-loaders may
                require categorical data to be separated from the numerical one.
                Specify categorical data here according to the specifications in
                the data loader.
            y: A dataframe containing labels data. Some data-loaders may require
                labels data to be separated from the rest. Specify labels data
                here according to the specifications in the data loader.
            settings: A dictionary containing custom parameters that can change
                between different data loaders.

        Returns:
            numpy.ndarray[numpy.float64[m, 1], flags.writeable, flags.f_contiguous]:
            A vector containing the predictions of the model.
        """
        data = mlconcepts.data.load(dataset, categorical=categorical,
                                    labels=labels, Xc=Xc, y=y, settings=settings)
        return self.model.predict(
            data.X if data.X is not None else 
                np.empty(shape=(0, 0), dtype=np.float64, order="F"),
            data.Xc if data.Xc is not None else 
                np.empty(shape=(0, 0), dtype=np.int32, order="F")
        )
    
    def predict_explain(self, dataset, categorical=[], labels=None, Xc=None, 
                        y=None, settings={}):
        """Predicts labels for the given dataset and returns explanation data.

        Args:
            dataset: A dataset represented in some format. The type/format of the
                dataset is automatically detected and a data-loader is used
                accordingly.
            categorical: A list of features suggested to be categorical. Data
                loaders should automatically detect categorical features. This
                parameter is used for those categorical features which are hard to
                distinguish from numerical ones, e.g., columns containing only 0 or
                1.
            labels: The name of the labels column in the dataset.
            Xc: A dataframe containing categorical data. Some data-loaders may
                require categorical data to be separated from the numerical one.
                Specify categorical data here according to the specifications in
                the data loader.
            y: A dataframe containing labels data. Some data-loaders may require
                labels data to be separated from the rest. Specify labels data
                here according to the specifications in the data loader.
            settings: A dictionary containing custom parameters that can change
                between different data loaders.

        Returns:
            :class:`mlconcepts.ExplanationData`:
            An object containing explanation data and predictions for the whole
            dataset.
        """
        data = mlconcepts.data.load(dataset, categorical=categorical,
                                    labels=labels, Xc=Xc, y=y, settings=settings)
        pred, outdegs = self.model.predict_explain(
            data.X if data.X is not None else 
                np.empty(shape=(0, 0), dtype=np.float64, order="F"),
            data.Xc if data.Xc is not None else 
                np.empty(shape=(0, 0), dtype=np.int32, order="F")
        )
        return ExplanationData(
            predictions = pred,
            outdegs = outdegs,
            dataset = data,
            feature_sets = self.model.get_feature_sets()
        )
    
    def estimate_size(self):
        """Estimates the size of the model in bytes.

        Returns:
            int: A (slightly lower) estimate of the size of the model.
        """
        return self.model.estimate_size()

    def save(self, filename):
        """Compresses the model and writes it into a file.

        Args:
            filename: Path to the file which will be written.
        """
        self.model.save(filename)

    def load(self, filename):
        """Loads and decompresses the model from a file.

        Args:
            filename: Path to the file which will be loaded.
        """
        self.model.load(filename)

    
