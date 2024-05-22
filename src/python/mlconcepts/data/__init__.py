"""Data module of the mlconcepts library.

This package takes care of importing data from common python data 
representation libraries and formats. This is achieved by using
data loaders, namely the type of each dataframe representation is
mapped to a function that handles its loading/conversion to the 
right format for the c++ module.

Datasets should be loaded using the function :func:`mlconcepts.data.load` which
takes care of selecting the correct data loader depending on the passed
information.

Todo:
    Once the c++ algorithms start supporting streaming dataset, 
    mlconcepts.data will support it, too.
"""
import pathlib
import numpy
from .Dataset import Dataset, basic_load
from .PathLoader import path_load
from .NumpyLoader import numpy_load

data_loaders = {}
"""
dict[type, function]: Maps supported dataset types to functions data load them,
reffered to as data loaders. Every data loader must have the same signature
as the function :func:`mlconcepts.data.load`.

To externally add a data loader, just update this dictionary.
"""


def load(dataset, categorical=[], labels=None, Xc=None, y=None, settings={}):
    """Loads a dataset to use within the mlconcepts library.
    
    Args:
        dataset: A dataset represented in some format. The type/format of the
            dataset is automatically detected and a data-loader is used
            accordingly.
        categorical (list): A list of features suggested to be categorical.
            Data loaders should automatically obvious detect categorical
            features, this should be used for those categorical features which
            are hard to distinguish from numerical ones, e.g., columns 
            containing only 0 or 1.
        labels: Suggests the name of the labels column in a dataset.
        Xc: A dataframe containing categorical data. Some data-loaders may
            require categorical data to be separated from the numerical one.
            In these cases, categorical should be specified here according to
            the specification in the dataloader.
        y: A dataframe containing labels data. Some data-loaders may require
            labels data to be separated from the rest. In these cases, 
            categorical should be specified here according to the
            specification in the dataloader.
        settings (dict, optional): A dictionary containing custom parameters 
            which can change between different data loaders.
    
    Returns:
        mlconcepts.data.Dataset: The dataset in the format required by the 
        mlconcepts library.
    """
    if isinstance(categorical, str):
        categorical = [ categorical ]
    if categorical is None:
        categorical = []
    settings["load"] = load
    if type(dataset) in data_loaders:
        return data_loaders[type(dataset)](dataset, categorical=categorical,
                                           labels=labels, Xc=Xc, y=y,
                                           settings=settings)
    raise ValueError("No available data loader for type " + str(type(dataset)))

data_loaders[Dataset] = basic_load
data_loaders[str] = path_load
data_loaders[pathlib.Path] = path_load
data_loaders[numpy.ndarray] = numpy_load
try:
    import pandas
    from .PandasLoader import pandas_load
    data_loaders[pandas.DataFrame] = pandas_load
except ImportError:
    pass