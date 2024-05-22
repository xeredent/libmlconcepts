"""Supervised outlier detection model.

This module implements supervised outlier detection models based on
`this paper <https://www.sciencedirect.com/science/article/pii/S0167923624000290>`_.

Examples:
    Fitting the model on a dataset::

        import mlconcepts

        model = mlconcepts.SODModel()
        model.fit("dataset.csv")

    The method :meth:`mlconcepts.SODModel.fit` accepts all the arguments that
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
            model = mlconcepts.SODModel(
                n = 32, 
                epochs = 1000, 
                show_training = False
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

class SODModel:
    """Represents a supervised model using FCA for outlier detection.
    
    This interface is a facade for `mlconceptscore`. Its usage is similar to
    the standard usage of `sklearn` models.
    """

    def __init__(self, n=32, quantizer="uniform", explorer="none",
                 singletons=True, doubletons=True, full=True,
                 learning_rate=0.01, momentum=0.01, stop_threshold=0.001,
                 epochs=2000, show_training=True):
        """Constructs a supervised outlier detection model.

        Args:
            n (int): The default number of bins for uniform quantization.
            quantizer (str): The quantizer for real features. Options are:

                * 'uniform': breaks the range of the feature into uniformly 
                  sized bins.
            explorer (str): Specifies the feature set exploration strategy. 
                Options are:

                * 'none': no exploration.
            singletons (bool): Whether singleton agendas should be generated.
            doubletons (bool): Whether doubleton agendas should be generated.
            full (bool): Whether the full agenda should be generated.
            learning_rate (float): Learning rate for gradient descent.
            momentum (float): Momentum for gradient descent.
            stop_threshold (float): Threshold for loss improvement under which
                training is interrupted.
            epochs (int): Number of epochs the model is trained for.
            show_training (bool): Whether to output information on training
                iterations.
        """
        self.model = None
        if quantizer == "uniform":
            if explorer == "none":
                self.model = mlconcepts.mlconceptscore.SODUniform(
                    n=n, singletons=singletons, doubletons=doubletons,
                    full=full, learningRate=learning_rate, momentum=momentum,
                    stopThreshold=stop_threshold, trainEpochs=epochs,
                    showTraining=show_training
                )
            else:
                raise ValueError("Invalid explorer. Options are ['none'].")
        else:
            raise ValueError("Invalid explorer. Options are ['uniform'].")

    def fit(self, dataset, categorical=[], labels=None, Xc=None, y=None,
            settings={}):
        """Trains the model on a dataset.

        Args:
            dataset: A dataset in some format. The format is detected 
                automatically using :func:`mlconcepts.data.load`.
            categorical: A list of features suggested to be categorical.
            labels: The name of the labels column in the dataset.
            Xc: A dataframe containing categorical data.
            y: A dataframe containing labels data.
            settings: A dictionary containing custom parameters.

        """
        data = mlconcepts.data.load(dataset, categorical=categorical,
                                    labels=labels, Xc=Xc, y=y,
                                    settings=settings)
        self.model.fit(data.X if data.X is not None else
                           np.empty(shape=(0, 0), dtype=np.float64, order="F"),
                       data.y,
                       data.Xc if data.Xc is not None else
                           np.empty(shape=(0, 0), dtype=np.int32, order="F"))

    def predict(self, dataset, categorical=[], labels=None, Xc=None, y=None,
                settings={}):
        """Predicts labels for a dataset.

        Args:
            dataset: A dataset in some format.
            categorical: A list of features suggested to be categorical.
            labels: The name of the labels column in the dataset.
            Xc: A dataframe containing categorical data.
            y: A dataframe containing labels data.
            settings: A dictionary containing custom parameters.

        Returns:
            numpy.ndarray[numpy.float64[m, 1], flags.writeable, flags.f_contiguous]:
            A vector containing the predictions of the model.
        """
        data = mlconcepts.data.load(dataset, categorical=categorical,
                                    labels=labels, Xc=Xc, y=y,
                                    settings=settings)
        return self.model.predict(
               data.X if data.X is not None else
                   np.empty(shape=(0, 0), dtype=np.float64, order="F"),
               data.Xc if data.Xc is not None else
                   np.empty(shape=(0, 0), dtype=np.int32, order="F")
        )

    def predict_explain(self, dataset, categorical=[], labels=None, Xc=None,
                        y=None, settings={}):
        """Predicts labels for a dataset and returns explanation data.

        Args:
            dataset: A dataset in some format.
            categorical: A list of features suggested to be categorical.
            labels: The name of the labels column in the dataset.
            Xc: A dataframe containing categorical data.
            y: A dataframe containing labels data.
            settings: A dictionary containing custom parameters.

        Returns:
            :class:`mlconcepts.ExplanationData`:
            An object containing explanation data and predictions for the whole
            dataset.
        """
        data = mlconcepts.data.load(dataset, categorical=categorical,
                                    labels=labels, Xc=Xc, y=y,
                                    settings=settings)
        pred, outdegs =  self.model.predict_explain(
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
            An estimate of the size of the model.
        """
        return self.model.estimate_size()

    def save(self, filename):
        """Compresses the model and writes it into a file.

        Args:
            filename: Path to the file to be written.
        """
        self.model.save(filename)

    def load(self, filename):
        """Loads the model from a file.

        Args:
            filename: Path to the file to be loaded.
        """
        self.model.load(filename)