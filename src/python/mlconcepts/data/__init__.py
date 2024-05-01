from .Dataset import Dataset, basic_load

#A dictionary mapping the type of a dataframe to its loader, i.e., a function
#mimicking the signature of the load function below.
data_loaders = {}


def load(dataset, categorical = [], labels = None, Xc = None, y = None, settings = {}):
    """
    Loads a dataset to use within the mlconcepts library.
    
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
    :return: The dataset in the format required by the mlconcepts library.
    :rtype: mlconcepts.data.Dataset
    """
    if categorical is None:
        categorical = []
    settings["load"] = load
    if type(dataset) in data_loaders:
        return data_loaders[type(dataset)](dataset, categorical = categorical, labels = labels, Xc = Xc,
                                                                              y = y, settings = settings)
    raise ValueError("No available data loader for type " + str(type(dataset)))

import pathlib
import numpy
from .PathLoader import path_load
from .NumpyLoader import numpy_load
data_loaders[Dataset] = basic_load
data_loaders[str] = path_load
data_loaders[pathlib.Path] = path_load
data_loaders[numpy.ndarray] = numpy_load
try:
    import pandas
    from .PandasLoader import pandas_load
    data_loaders[pandas.core.frame.DataFrame] = pandas_load
except ImportError:
    pass