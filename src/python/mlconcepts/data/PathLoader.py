from pathlib import Path
import numpy as np
from .Dataset import Dataset

def path_load(dataset, categorical = [], labels = None, Xc = None, y = None, settings = {}):
    """
    Loads a dataset to use within the mlconcepts library from a file.
    
    :param dataset: A path to the file.
    :param categorical: A list of features suggested to be categorical.
    :param Xc: Optional path to a file containing the categorical part of the dataset.
    :param y: Optional path to a file containing the labels part of the dataset.
    :param settings: Depends on what file format is read.
    For .xlsx, .csv, .json, and .sql files it supports all the optional parameters of the 
    corresponding pandas reading functions. These parameters should be inserted in a 
    dictionary called 'parse'.
    For .mat files, the parameters 'Xname', 'Xcname', 'yname', which default to 'X', 'Xc', and 'y', 
    respectively, indicate the names of the matlab matrices from which the corresponding parts
    of the dataset are extracted.
    :rtype: mlconcepts.data.Dataset
    :raises OSError: If the file does not exist.
    """
    dataset_path = Path(dataset)
    load = settings["load"]

    #checks whether the file exists and raises a clear exception
    if not dataset_path.exists():
        raise OSError("the file %s does not exist" % str(dataset_path))

    #load csv, xlsx, json, and sql using pandas
    if dataset_path.suffix in [".csv", ".xlsx", ".json", ".sql"]:
        try:
            import pandas
            parsemap = {".csv" : pandas.read_csv, ".xlsx" : pandas.read_excel, 
                        ".json" : pandas.read_json, ".sql" : pandas.read_json }
            parse_settings = settings["parse"] if "parse" in settings else {}
            df = parsemap[dataset_path.suffix](dataset, **parse_settings)
            return load(df, categorical = categorical, labels = labels, Xc = Xc, y = y, settings = settings)
        except ImportError:
            raise RuntimeError("The file formats csv, xlsx, json, and sql are supported only via pandas")
    
    #load matlab files
    if dataset_path.suffix == ".mat":
        #the matrices default names are X, Xc, and y, but they can be set in the settings
        Xname = settings["Xname"] if "Xname" in settings else "X"
        Xcname = settings["Xcname"] if "Xcname" in settings else "Xc"
        yname = settings["yname"] if "yname" in settings else "y"
        #By default load with scipy
        try:
            import scipy.io
            mat = scipy.io.loadmat(dataset_path)
            X = np.asfortranarray(mat[Xname]).astype(np.float64) if Xname in mat else None
            Xc = np.asfortranarray(mat[Xcname]).astype(np.int32) if Xcname in mat else None
            y = np.asfortranarray(mat[yname]).astype(np.int32) if yname in mat else None
            return Dataset(X, Xc, y)
        except (ImportError, NotImplementedError):
            try: #If scipy is not installed, or the version is not supported, fall back to h5py
                import h5py
                with h5py.File(dataset_path, 'r') as mat:
                    X = np.array(mat[Xname], dtype = np.float64, order = "C").transpose() if Xname  in mat else None
                    Xc = np.array(mat[Xcname], dtype = np.int32, order = "C").transpose() if Xcname in mat else None
                    y  = np.array(mat[yname ], dtype = np.int32, order = "C").transpose() if yname  in mat else None
                    return Dataset(X, Xc, y)
            except ImportError:
                raise RuntimeError("Matlab file support requires scipy for versions < 7.3, and h5py for versions >= 7.3")
        pass
