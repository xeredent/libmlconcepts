"""Implements a data loader for pandas dataframes."""

import pandas as pd
import numpy as np
from .Dataset import Dataset

def pandas_load(dataset, categorical=[], labels=None, Xc=None, y=None, settings={}):
    """Loads a pandas dataset to use within the mlconcepts library.

    This function should not be called directly, but rather indirectly via
    :func:`mlconcepts.data.load`.
    
    Args:
        dataset (pandas.core.frame.DataFrame): A pandas DataFrame.
        categorical (list-like[str]): A list of features suggested to be
            categorical. Numerical features in the list will be treated as
            categorical. It is not necessary to pass also non-numerical
            features in this parameter.
        labels (str): The name of the column containing the labels.
        Xc: Ignored.
        y (numpy.ndarray): Optional numpy array containing the labels column.
            Overrides any column dataset[labels]. Must be one-dimensional and
            its elements must be convertible to int32.
        settings: Ignored.

    Returns:
        :class:`mlconcepts.data.Dataset`: A dataset in the right format for
        mlconcepts algorithms.
    """
    #Check optional parameters
    if labels is not None and not isinstance(labels, str):
        raise TypeError("labels should be the name of the column containing "
                        "the labels, found type " + str(type(labels)))
    if labels is not None and labels not in dataset.columns:
        raise ValueError("Column " + labels + " not found in dataset")
    
    # Filter out numerical data, and possibly exclude the labels column
    data_num = dataset.select_dtypes(include="number")
    data_labels = None
    if y is not None:  # y gets priority over the labels column
        data_labels = np.asfortranarray(y).astype(np.int32)
    elif labels is not None and labels in dataset:
        if labels in data_num:
            data_labels = dataset[labels]
            data_num = data_num.drop(labels, axis=1)
        else:
            data_labels = pd.factorize(dataset[labels])[0]

    # Filter out categorical data and factorize it
    cat_features = list(filter(
                 lambda c: ((c in categorical or dataset[c].dtype == "O")
                            and 
                            c != labels), 
                 dataset.columns
    ))
    data_cat = pd.DataFrame(
             {f: pd.factorize(dataset[f])[0] for f in cat_features}
    )

    # Drop forced categorical features from the numerical part of the dataset
    data_num.drop(
        [f for f in categorical if f in data_num.columns], 
        axis = 1,
        inplace = True
    )
    
    # Create dataset object and set feature names
    data = Dataset(
        None if data_num.empty else
            np.asfortranarray(data_num.to_numpy().astype(np.float64)),
        None if data_cat.empty else
            np.asfortranarray(data_cat.to_numpy().astype(np.int32)),
        None if data_labels is None else
            np.asfortranarray(data_labels.astype(np.int32))
    )
    data.set_real_names(data_num.columns)
    data.set_categorical_names(cat_features)
    data.set_labels_name(labels)
    return data
