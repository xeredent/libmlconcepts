import mlconceptscore
import mlconcepts.data
import numpy as np

class SODModel:
    """
    Represents a supervised model that uses formal concept analysis and interrogative agendas
    to achieve explainable outlier detection.
    """

    def __init__(self, n = 32, quantizer = "uniform", explorer = "none", 
                 singletons = True, doubletons = True, full = True,
                 learning_rate = 0.01, momentum = 0.01, stop_threshold = 0.001,
                 epochs = 2000, show_training = True):
        """
        Constructs a supervised outlier detection model.
        :param int n: The default number of bins for uniform quantization.
        :param str quantizer: The quantizer for real features. The available options are:
            - 'uniform' for a quantizer that breaks the range of the feature into uniformly sized bins.
        :param str explorer: Specifies the feature set exploration strategy. The available options are:
            - 'none' for no exploration.
        :param bool singletons: Whether singleton agendas should be generated.
        :param bool doubletons: Whether doubleton agendas should be generated.
        :param bool full: Whether the full agenda should be generated.
        :param float learning_rate: The learning rate for gradient descent.
        :param float momentum: The momentum for gradient descent.
        :param float stop_threshold: The threshold for loss improvement under which training is interrupted.
        :param int epochs: The number of epochs the model is trained for.
        :param bool show_training: Whether to output in the standard output information on the training iterations.
        """
        self.model = None
        if quantizer == "uniform":
            if explorer == "none":
                self.model = mlconceptscore.SODUniform(n = n, singletons = singletons, doubletons = doubletons, full = full,
                             learningRate = learning_rate, momentum = momentum, stopThreshold = stop_threshold, 
                             trainEpochs = epochs, showTraining = show_training)
            else:
                raise ValueError("The only available value for the explorer is 'none'")
        else:
            raise ValueError("The only available value for the quantizer is 'uniform'")

    def fit(self, dataset, categorical = [], labels = None, Xc = None, y = None, settings = {}):
        """
        Trains the model on some dataset. The dataset should contain labelled data.
        
        :param dataset: A dataset represented in some format. The type/format of the dataset is automatically
        detected and a data-loader is used accordingly.
        :param categorical: A list of features suggested to be categorical. Data loaders should automatically 
        obvious detect categorical features, this should be used for those categorical features which are hard
        to distinguish from numerical ones, e.g., columns containing only 0 or 1.
        :param labels: Suggests the name of the labels column in a dataset.
        :param Xc: A dataframe containing categorical data. Some data-loaders may require categorical data to be
        separated from the numerical one. In these cases, categorical should be specified here according to the
        specification in the dataloader.
        :param y: A dataframe containing labels data. Some data-loaders may require labels data to be
        separated from the rest. In these cases, categorical should be specified here according to the
        specification in the dataloader.
        :param settings: A dictionary containing custom parameters which can change between different data loaders.
        """
        data = mlconcepts.data.load(dataset, categorical = categorical, labels = labels, 
                                    Xc = Xc, y = y, settings = settings)
        self.model.fit(data.X if data.X is not None else np.empty(shape = (0, 0), dtype = np.float64, order = "F"),
                       data.y,
                       data.Xc if data.Xc is not None else np.empty(shape = (0, 0), dtype = np.int32, order = "F"))

    def predict(self, dataset, categorical = [], labels = None, Xc = None, y = None, settings = {}):
        """
        Trains the model on some dataset. The dataset should contain labelled data.
        
        :param dataset: A dataset represented in some format. The type/format of the dataset is automatically
        detected and a data-loader is used accordingly.
        :param categorical: A list of features suggested to be categorical. Data loaders should automatically 
        obvious detect categorical features, this should be used for those categorical features which are hard
        to distinguish from numerical ones, e.g., columns containing only 0 or 1.
        :param labels: Suggests the name of the labels column in a dataset.
        :param Xc: A dataframe containing categorical data. Some data-loaders may require categorical data to be
        separated from the numerical one. In these cases, categorical should be specified here according to the
        specification in the dataloader.
        :param y: A dataframe containing labels data. Some data-loaders may require labels data to be
        separated from the rest. In these cases, categorical should be specified here according to the
        specification in the dataloader.
        :param settings: A dictionary containing custom parameters which can change between different data loaders.
        :return: A vector containing the predictions of the model.
        """
        data = mlconcepts.data.load(dataset, categorical = categorical, labels = labels, 
                                    Xc = Xc, y = y, settings = settings)
        return self.model.predict(data.X if data.X is not None else np.empty(shape = (0, 0), dtype = np.float64, order = "F"),
                           data.Xc if data.Xc is not None else np.empty(shape = (0, 0), dtype = np.int32, order = "F"))
    
    def estimate_size(self):
        """
        Estimates the size of the model in bytes.
        :return: A (slightly lower) estimate of the size of the model.
        :rtype int:
        """
        return self.model.estimate_size()

    def save(self, filename):
        """
        Compresses the model and writes it into a file.
        :param filename: Path to the file which will be written.
        """
        self.model.save(filename)

    def load(self, filename):
        """
        Loads the model from a file.
        :param filename: Path to the file which will be loaded.
        """
        self.model.load(filename)

    
