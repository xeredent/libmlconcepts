import numpy

class Dataset(object):
    """
    Abstract representation of a dataset.
    This library provides functions to convert the dataframes from several libraries into datasets of this class.
    The constructors of this class implements checks to make sure that the (possibly external) data loader satisfies
    all the format restrictions of the core library.

    A Dataset object consists of three main fields:
        - X, a numpy array such that dtype=float64 and flags=f_contiguous.
        - Xc, a numpy array such that dtype=int32 and flags=f_contiguous.
        - y, a numpy one dimensional array such that dtype=int32 and flags=f_contiguous.
    One of X and Xc can be empty. Also y can be empty.
    The size of y must match that of the rows of X and Xc.
    """

    def __init__(self, X = None, Xc = None, y = None):
        self.X = X
        self.Xc = Xc
        self.y = y
        if X is None and Xc is None:
            raise ValueError("At least one of the numerical and categorical parts of a dataset must be non-empty.")
        if X is not None:
            if type(X) != numpy.ndarray:
                raise ValueError("The numerical part of a dataset must be a numpy.ndarray.")
            if X.dtype != "float64":
                raise ValueError("The numerical part of a dataset must be of dtype float64.")
            if X.flags.f_contiguous == False:
                raise ValueError("The numerical part of a dataset must be such that flags.f_contiguous is True.")
        if Xc is not None:
            if type(Xc) != numpy.ndarray:
                raise ValueError("The categorical part of a dataset must be a numpy.ndarray.")
            if Xc.dtype != "int32":
                raise ValueError("The categorical part of a dataset must be of dtype int32.")
            if Xc.flags.f_contiguous == False:
                raise ValueError("The categorical part of a dataset must be such that flags.f_contiguous is True.")
        if y is not None:
            if type(y) != numpy.ndarray:
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
        Sets the names of the real features
        :param names: A dictionary mapping a feature name to its column index in self.X, or a collection
        indexed by integers as long as the number of columns of self.X.
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
        Sets the names of the categorical features
        :param names: A dictionary mapping a feature name to its column index in self.Xc, or a collection
        indexed by integers as long as the number of columns of self.Xc.
        :raises ValueError: If names is of the wrong type.
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

    def set_label_name(self, name):
        """
        Sets the name of the label
        :param name: The name of the label feature. If None, an empty string is inserted instead.
        :raises ValueError: If name is of the wrong type.
        """
        if type(name) is not str and name is not None:
            raise ValueError("the label name must be a string, found " + str(type(name)))
        self.yname = name if name is not None else ""



    def size(self):
        """
        Retrieves the size of the dataset.
        :return: The size of the dataset.
        """
        if self.X is not None:
            return self.X.shape[0]
        return self.Xc.shape[0]



def basic_load(dataset, categorical = [], labels = None, Xc = None, y = None, settings = {}):
    """
    Dummy dataset loader.
    
    :param dataset: A dataset of type mlconcepts.data.Dataset.
    :param categorical: Ignored.
    :param labels: Ignored.
    :param Xc: Ignored.
    :param y: Ignored.
    :param settings: Ignored.
    :return: The dataset in the format required by the mlconcepts library.
    :rtype: mlconcepts.data.Dataset
    """
    return dataset