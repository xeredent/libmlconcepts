import numpy as np

class Dataset(object):
    """
    Abstract representation of a dataset.
    This library provides functions to convert the dataframes from several libraries into datasets of this class.
    The constructors of this class implements checks to make sure that the (possibly external) data loader satisfies
    all the format restrictions of the core library.

    :ivar X: The numerical part of the dataset. Can be null if a categorical part is present.
        The objects are encoded row-wise, and the features column-wise.
        The number of rows of ``X`` must match those of ``Xc``, if present.
    :vartype X: numpy.ndarray[numpy.float64[n, m], flags.writeable, flags.f_contiguous]
    :ivar Xc: The categorical part of the dataset. Can be null if a numerical part is present.
        The objects are encoded row-wise, and the features column-wise.
        The number of rows of ``Xc`` must match those of ``X``, if present.
    :vartype Xc: numpy.ndarray[numpy.int32[n, m], flags.writeable, flags.f_contiguous]
    :ivar y: The labels of the dataset. Can be null.
    :vartype y: numpy.ndarray[numpy.float64[n, 1], flags.writeable, flags.f_contiguous]
    """

    def __init__(self, X = None, Xc = None, y = None):
        self.X = X
        self.Xc = Xc
        self.y = y
        if X is None and Xc is None:
            raise ValueError("At least one of the numerical and categorical parts of a dataset must be non-empty.")
        if X is not None:
            if type(X) != np.ndarray:
                raise ValueError("The numerical part of a dataset must be a numpy.ndarray.")
            if X.dtype != "float64":
                raise ValueError("The numerical part of a dataset must be of dtype float64.")
            if X.flags.f_contiguous == False:
                raise ValueError("The numerical part of a dataset must be such that flags.f_contiguous is True.")
        if Xc is not None:
            if type(Xc) != np.ndarray:
                raise ValueError("The categorical part of a dataset must be a numpy.ndarray.")
            if Xc.dtype != "int32":
                raise ValueError("The categorical part of a dataset must be of dtype int32.")
            if Xc.flags.f_contiguous == False:
                raise ValueError("The categorical part of a dataset must be such that flags.f_contiguous is True.")
        if y is not None:
            if type(y) != np.ndarray:
                raise ValueError("The labels part of a dataset must be a numpy.ndarray.")
            if y.dtype != "int32":
                raise ValueError("The labels part of a dataset must be of dtype int32.")
            if y.flags.f_contiguous == False:
                raise ValueError("The labels part of a dataset must be such that flags.f_contiguous is True.")
        if X is not None and Xc is not None and X.shape[0] != Xc.shape[0]:
            raise ValueError("The number of rows of the numerical and categorical part of a dataset must match.")
        if y is not None and y.shape[0] != self.size():
            raise ValueError("The number of labels in a dataset must match the size of the dataset")
        self.Xnames = ["" for x in range(self.X.shape[1])] if self.X is not None else []
        self.Xcnames = ["" for x in range(self.Xc.shape[1])] if self.Xc is not None else []
        self.Xnamedict = {}
        self.Xcnamedict = {}
        self.yname = ""
        
    
    def set_real_names(self, names):
        """
        Sets the names of the real features.

        :param names: A dictionary mapping a feature name to its column index in self.X, or a collection
            indexed by integers as long as the number of columns of self.X.
        :type names: list[str] or dict[str,int]

        :raises ValueError: If names is of the wrong type.
        :raises RuntimeError: If one of the indices is larger than the number of columns of self.X, or, 
            if names is a list, if it is smaller than the number of columns of X. It is not guaranteed that
            the internal state of the object has not changed.
        """
        if self.X is None:
            return
        if type(names) is dict:
            self.Xnamedict = {}
            self.Xnames = ["" for x in range(self.X.shape[1])]
            for k in names:
                if type(names[k]) is not int:
                    raise RuntimeError("the range of names must contain only integers, found " + str(type(names[k])))
                if names[k] < self.X.shape[1]:
                    self.Xnames[k] = v
                else:
                    raise RuntimeError("index %d is larger than the number of features of X (%d)" % (v, self.X.shape[1]))
        else:
            self.Xnamedict = {}
            self.Xnames = ["" for x in range(self.X.shape[1])]
            for i in range(len(names)):
                if type(names[i]) is not str:
                    raise RuntimeError("the list of names must contain only strings, found " + str(type(names[i])))
                self.Xnames[i] = names[i]
                self.Xnamedict[names[i]] = i
    
    def set_categorical_names(self, names):
        """
        Sets the names of the categorical features.

        :param names: A dictionary mapping a feature name to its column index in self.Xc, or a collection
            indexed by integers as long as the number of columns of self.Xc.
        :type names: list[str] or dict[str,int]

        :raises RuntimeError: If one of the indices is larger than the number of columns of self.Xc, or, 
            if names is a list, if it is smaller than the number of columns of Xc. It is not guaranteed that
            the internal state of the object has not changed.
        """
        if self.Xc is None:
            return
        if type(names) is dict:
            self.Xcnamedict = {}
            self.Xcnames = ["" for x in range(self.Xc.shape[1])]
            for k in names:
                if type(names[k]) is not int:
                    raise RuntimeError("the range of names must contain only integers, found " + str(type(names[k])))
                if names[k] < self.Xc.shape[1]:
                    self.Xcnames[k] = v
                else:
                    raise RuntimeError("index %d is larger than the number of features of Xc (%d)" % (v, self.X.shape[1]))
        else:
            self.Xcnamedict = {}
            self.Xcnames = ["" for x in range(self.Xc.shape[1])]
            for i in range(len(names)):
                if type(names[i]) is not str:
                    raise RuntimeError("the list of names must contain only strings, found " + str(type(names[i])))
                self.Xcnames[i] = names[i]
                self.Xcnamedict[names[i]] = i

    def set_labels_name(self, name):
        """
        Sets the name of the label

        :param name: The name of the label feature. If None, an empty string is inserted instead.
        :type name: str

        :raises TypeError: If name is of the wrong type.
        """
        if type(name) is not str and name is not None:
            raise TypeError("the label name must be a string, found " + str(type(name)))
        self.yname = name if name is not None else ""


    def size(self):
        """
        Retrieves the size of the dataset.

        :returns: The size of the dataset.
        :rtype: int
        """
        if self.X is not None:
            return self.X.shape[0]
        return self.Xc.shape[0]

    def split(self, splitter):
        """
        Generates train-test splits as specified by the input splitter.

        :param splitter: An object implementing  a method split(X, y) or split(X), to which
            the generation of the index-sets for the train-test splits is relayed. 
            At each iteration yields two Dataset objects representing the train and test set, respectively.
        """
        #Creates the generator from the splitter. If y is null, it calls split(X), otherwise split(X, y).
        #Also checks whether one of X and Xc is null, and picks one of the two to generate the index sets.
        #Both the datasets will be splitted in the output.
        generator = (splitter.split(self.X, self.y) if self.X is not None else splitter.split(self.Xc, self.y))\
                    if self.y is not None else \
                    (splitter.split(self.X) if self.X is not None else splitter.split(self.Xc)) 
        for train_indices, test_indices in generator:
            #Split the matrices
            x_train  = np.asfortranarray(self.X[train_indices]) if self.X  is not None else None
            xc_train = np.asfortranarray(self.Xc[train_indices]) if self.Xc is not None else None
            y_train  = np.asfortranarray( self.y[train_indices]) if self.y  is not None else None
            x_test  =  np.asfortranarray(self.X[test_indices]) if self.X  is not None else None
            xc_test = np.asfortranarray(self.Xc[test_indices]) if self.Xc is not None else None
            y_test  = np.asfortranarray( self.y[test_indices]) if self.y  is not None else None
            #And create the datasets
            train = Dataset(X = x_train, Xc = xc_train, y = y_train)
            test = Dataset(X = x_test, Xc = xc_test, y = y_test)
            #Transfer the names to the splits
            train.set_real_names(self.Xnames); test.set_real_names(self.Xnames)
            train.set_categorical_names(self.Xcnames); test.set_categorical_names(self.Xcnames)
            train.set_labels_name(self.yname); test.set_labels_name(self.yname)
            yield train, test




def basic_load(dataset, categorical = [], labels = None, Xc = None, y = None, settings = {}):
    """
    Dummy dataset loader.
    
    :param dataset: A dataset which is returned as is.
    :type dataset: mlconcepts.data.Dataset
    :param categorical: Ignored.
    :param labels: Ignored.
    :param Xc: Ignored.
    :param y: Ignored.
    :param settings: Ignored.

    :returns: The dataset in the format required by the mlconcepts library.
    :rtype: mlconcepts.data.Dataset
    """
    return dataset