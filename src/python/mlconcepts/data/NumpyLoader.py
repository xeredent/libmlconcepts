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
        of ``dataset`` if ``dataset`` is not empty.
    :type Xc: numpy.ndarray
    :param y: optional Optional numpy array containing the labels column. Its dtype must be convertible to int32.
    :type y: numpy.ndarray
    :param settings: Ignored.

    :returns: A dataset in the right format for mlconcepts algorithms.
    :rtype: :class:`mlconcepts.data.Dataset`
    """
    return Dataset(np.asfortranarray(dataset.astype(np.float64)) if dataset is not None else None, 
                   np.asfortranarray(Xc.astype(np.int32)) if Xc is not None else None, 
                   y.astype(np.int32) if y is not None else None)
