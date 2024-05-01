import numpy as np
from .Dataset import Dataset

def numpy_load(dataset, categorical = [], labels = None, Xc = None, y = None, settings = {}):
    """
    Loads a numpy array to use within the mlconcepts library.
    
    :param dataset: A numpy.ndarray whose elements can be converted to float64. Represents the
    numerical part of the dataset.
    :param categorical: Ignored.
    :param labels: Ignored.
    :param Xc: Optional numpy.ndarray whose elements can be converted to int32. Its number of rows must match that
    of 'dataset' if 'dataset' is not empty.
    :param y: Optional numpy array containing the labels column. Its dtype must be convertible to int32.
    :param settings: Ignored.
    :rtype: mlconcepts.data.Dataset
    """
    return Dataset(dataset, Xc, y)
