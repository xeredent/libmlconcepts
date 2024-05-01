import pandas
import numpy as np
from .Dataset import Dataset

def pandas_load(dataset, categorical = [], labels = None, Xc = None, y = None, settings = {}):
    """
    Loads a pandas dataset to use within the mlconcepts library.
    
    :param dataset: A pandas DataFrame.
    :param categorical: A list of features suggested to be categorical. Numerical features in the list will
    be treated as categorical. It is not necessary to pass also non-numerical features in this parameter.
    :param labels: The name of the column containing the labels.
    :param Xc: Should be None (ignored).
    :param y: Optional numpy array containing the labels column. Overrides any column dataset[labels].
    :param settings: Ignored.
    :rtype: mlconcepts.data.Dataset
    """
    #Check optional parameters
    if labels is not None and type(labels) != str:
        raise TypeError("labels should be the name of the column containing the labels")
        if labels not in dataset.columns:
            raise ValueError("Column " + labels + " not found in dataset")
    
    #Filter out numerical data, and possibly exclude the labels column
    data_num = dataset.select_dtypes(include="number")
    data_labels = None
    if y is not None: #y gets priority over the labels column
        data_labels = np.asfortranarray(y).astype(np.int32)
    elif labels is not None and labels in dataset:
        if labels in data_num:
            data_labels = dataset[labels]
        else:
            data_labels = pandas.factorize(dataset[labels])[0]
        data_num = data_num.drop([labels], axis = 1)

    #Filter out categorical data and factorize it
    cat_features = list(filter(lambda c: (c in categorical or dataset[c].dtype == "O") and c != labels, dataset.columns))
    data_cat = pandas.DataFrame({f : pandas.factorize(dataset[f])[0] for f in cat_features})
    
    

    #Create dataset object and set feature names
    data = Dataset( np.asfortranarray(data_num.to_numpy().astype(np.float64)),
             np.asfortranarray(data_cat.to_numpy().astype(np.int32)),
             np.asfortranarray(data_labels.astype(np.int32)) if data_labels is not None else None )
    data.set_real_names(data_num.columns)
    data.set_categorical_names(cat_features)
    data.set_label_name(labels)
    return data