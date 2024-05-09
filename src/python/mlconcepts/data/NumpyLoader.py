"""Implements a data loader for numpy matrices."""

import numpy as np
from .Dataset import Dataset

def numpy_load(dataset, categorical=[], labels=None, Xc=None, y=None,
               settings={}):
    """Loads a numpy array to use within the mlconcepts library.

    This function should not be called directly, but rather indirectly via
    :func:`mlconcepts.data.load`.
    
    Args:
        dataset (numpy.ndarray): A numpy array whose elements can be converted
            to float64. Represents the numerical part of the dataset.
        categorical (list): Ignored.
        labels: Ignored.
        Xc (numpy.ndarray): Optional numpy array whose elements can be 
            converted to int32. Its number of rows must match that of 
            ``dataset`` if ``dataset`` is not empty.
        y (numpy.ndarray): Optional numpy array containing the labels column.
            Its dtype must be convertible to int32.
        settings: Ignored.

    Returns:
        :class:`mlconcepts.data.Dataset`: A dataset in the right format for
        mlconcepts algorithms.
    """
    return Dataset(
           None if dataset is None else
               np.asfortranarray(dataset.astype(np.float64)),
           None if Xc is None else
               np.asfortranarray(Xc.astype(np.int32)), 
           None if y is None else
               y.astype(np.int32)
    )
