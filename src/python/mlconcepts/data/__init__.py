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
from .dataset import Dataset, basic_load, GraphDataset
from .path_load import path_load
from .numpy_load import numpy_load

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
    from .pandas_load import pandas_load
    data_loaders[pandas.DataFrame] = pandas_load
except ImportError:
    pass

def graph_load_adgad(network=None, attributes=None, classes=None):
    """Loads a graph dataset following the format used by ADGAD.

    Args:
        network (scipy.sparse): the adjacency matrix of the graph. Its cells
            must contain either 0 or 1.
        attributes (scipy.sparse): a matrix containing the attributes of each
            node. This matrix has as many rows as the nodes, and as many 
            columns as their attributes. Its dtype must be convertible to 
            float64.
        classes (numpy.ndarray): a vector containing the class of each node in
            the graph. Its dtype must be convertible to int32.

    Returns:
        mlconcepts.data.GraphDataset: a GraphDataset object encoding the graph
        dataset.
    """
    sources, targets = network.nonzero()
    return GraphDataset(
        nodes = load(attributes.todense()),
        sources = sources,
        targets = targets,
        y = classes
    )

def graph_load_adgad_file(filepath="", network_name="Network",
                          attributes_name="Attributes", class_name='Class'):
    """Loads a graph dataset following the format used by ADGAD.

    Datasets in the format of ADGAD (Awesome Deep Graph Anomaly Detection) are
    .mat files defining three sparse matrices:
    - "Network", the adjacency matrix of the graph. Its cells must contain
      either 0 or 1.
    - "Attributes", a matrix containing the attributes of each node. This
      matrix has as many rows as the nodes, and as many columns as their 
      attributes. Its dtype must be convertible to float64.
    - "Class", a vector containing the class of each node in the graph. Its
      dtype must be convertible to int32.

    Args:
        filepath (str or Path): A path to a mat file defining the three
            required matrices.
        network_name (str): the name of the network matrix. Default "Network".
        attributes_name (str): the name of the attributes matrix. Defaults to
            "Attributes".
        class_name (str): the name of the class matrix. Defaults to "Class".

    Returns:
        mlconcepts.data.GraphDataset: a GraphDataset object encoding the graph
        dataset.

    Throws:
        ModuleNotFoundError: if scipy is not installed in the machine.
    """
    try:
        import scipy.io
        m = scipy.io.loadmat(filepath)
        return graph_load_adgad(
            network=m[network_name], 
            attributes=m[attributes_name],
            classes=m[class_name]
        )
    except ModuleNotFoundError:
        raise ModuleNotFoundError("install scipy to use this function")

def graph_load_elliptic(features=None, edges=None, classes=None,
                        features_tx_name=None, classes_tx_name=None,
                        class_name=None, src_name=None, dst_name=None,
                        categorical=[]):
    """Loads a dataset in the format of the elliptic dataset.

    This function requires `pandas` to be installed.

    Args:
        features (str or Path or pandas.DataFrame): A (path to) a dataset
            containing the features of the nodes of the graph. Cannot be None.
        edges (str or Path or pandas.DataFrame): A (path to) a dataset
            containing the edges of the dataset. Cannot be None.
        classes (str or Path or pandas.DataFrame): A (path to) a dataset
            containing the classes of the nodes in the dataset. This dataset
            can coincide with the "features" dataset, and, if it does, the
            column indicating the classes is removed from the features dataset.
            Cannot be None.
        features_tx_name (str): The name of the column containing the ID of the
            node in the features dataset. If it is None, it is assumed that the
            ID coincides with the index of the row in the dataset.
        classes_tx_name (str): The name of the column containing the ID of the
            node in the classes dataset. If it is None, it is assumed that the
            ID coincides with the index of the row in the dataset. It should be
            None if and only if also features_tx_name is None.
        class_name (str): The name of the column containing the class
            of the node in the classes dataset. Cannot be None.
        src_name (str): The ID of the column indicating the ID of the
            source nodes in the edges dataset. It cannot be None.
        dst_name (str): The ID of the column indicating the ID of the
            destination nodes in the edges dataset. It cannot be None.
        categorical (list[str]): A list of columns in the features dataset
            which are considered to be categorical.
    
    Returns:
        mlconcepts.data.GraphDataset: An object encoding the graph structure
        and the features of the nodes.
    
    Raises:
        ModuleNotFoundError: if `pandas` is not installed.
    """
    import pandas as pd
    assert edges is not None, "the edges must be specified"
    assert features is not None, "the features of the nodes must be specified"
    assert classes is not None, "the classes of the nodes must be specified"
    assert class_name is not None, "the class column must be specified"
    assert src_name is not None and dst_name is not None
    assert ((features_tx_name is None and classes_tx_name is None) or
           (features_tx_name is not None and classes_tx_name is not None))
    features = (features if isinstance(features, pd.DataFrame) 
                else pd.read_csv(features))
    edges = (edges if isinstance(edges, pd.DataFrame)
             else pd.read_csv(edges))
    classes = (classes if isinstance(classes, pd.DataFrame) 
               else pd.read_csv(classes))
    # If the nodes have some IDs in their data, use them to encode the edges
    if features_tx_name is not None:
        unique = numpy.unique(features[features_tx_name])
        factors = numpy.arange(len(unique))
        edges[[src_name, dst_name]] = edges[[src_name, dst_name]].replace(
            unique, 
            factors
        )
        if classes is not features:
            classes[classes_tx_name] = classes[classes_tx_name].replace(
                unique, 
                factors
            )
            classes.sort_values(classes_tx_name, inplace=True)
            classes.drop(classes_tx_name, axis=1, inplace=True)
        features.sort_values(features_tx_name, inplace=True)
        features.drop(features_tx_name, axis=1, inplace=True)
    y = classes[class_name].to_numpy(dtype=numpy.int32)
    classes.drop(class_name, axis=1, inplace=True)
    return GraphDataset(
        nodes=load(features, categorical=categorical), 
        sources=edges[src_name].to_numpy(dtype=numpy.int32),
        targets=edges[dst_name].to_numpy(dtype=numpy.int32),
        y=y
    )