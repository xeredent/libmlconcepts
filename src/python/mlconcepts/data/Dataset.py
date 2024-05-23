""":class:`Dataset` objects hold data to be used by the mlconcepts package.

Data can be imported from common python data 
representation libraries and formats. This is achieved by using
data loaders, namely the type of each dataframe representation is
mapped to a function that handles its loading/conversion to the 
right format for the c++ module.

Datasets should be loaded using the function :func:`mlconcepts.data.load` which
takes care of selecting the correct data loader depending on the passed
information.

Examples:
    The following code snippets show how to load datasets formats.
    We assume that our data has two numerical features, "feature1" and 
    "feature2", and a cateogorical feature "color". 

    *Pandas*. The type of the feature in pandas is automatically inferred:
    ::

        import pandas
        import mlconcepts.data

        df = pandas.DataFrame({
            "feature1" : [3.5, 1.6],
            "feature2" : [1.6, 2.2],
            "color" : ["blue", "green"],
            "classes"  : ["class1", "class2"]
        })
        data = mlconcepts.data.load(df, labels = "classes")

    *Numpy*:
    ::

        import numpy as np
        import mlconcepts.data

        data = mlconcepts.data.load(
            np.array([[3.5, 5.3], [1.6, 2.2]]), 
            Xc = np.array([[0], [1]]),
            y = np.array([0, 1])
        )

    *Files*. The following example assumes that in the working directory the
    files "dataset.csv", "dataset.xlsx", "dataset.json", "dataset.sql", and
    "dataset.mat" (Matlab file format) can be found. We assume that the first
    four contain a column called "classes" to label the class of the entries.
    ::

        import mlconcepts.data

        data = mlconcepts.data.load("dataset.csv", labels = "classes")
        data = mlconcepts.data.load("dataset.xlsx", labels = "classes")
        data = mlconcepts.data.load("dataset.json", labels = "classes")
        data = mlconcepts.data.load("dataset.sql", labels = "classes")
        data = mlconcepts.data.load(
            "dataset.mat", 
            settings = {
                "Xname" : "X" # X is the default name, this line can be skipped
                "yname" : "y" # y is the default name, this line can be skipped
            }
        )


Todo:
    Once the c++ algorithms start supporting streaming dataset, 
    mlconcepts.data.Dataset will support it, too.
"""

import numpy as np

class Dataset(object):
    """Representation of dataset used by the mlconcepts library.
    
    Dataset consist of possibly three parts, namely a numerical part,
    a categorical part, and labels for each entry. This class ensures that
    the format required by the core library is used by ``mlconcepts``
    objects. You should not initialize Dataset objects on your own, unless
    you are implementing a data loader (see :mod:`mlconcepts.data`).

    Datasets should be loaded using the function :func:`mlconcepts.data.load`.

    Future versions of this library will support streaming datasets.
       
    Attributes:
        X (numpy.ndarray): Numerical part of the dataset. Possibly ``None``.
        Xc (numpy.ndarray): Categorical part of the dataset. Possibly ``None``.
        y (numpy.ndarray): Labels of the dataset. Possibly ``None``.
    """

    def __init__(self, X=None, Xc=None, y=None):
        """Initializes a Dataset object.

        Args:
            X (:obj:`numpy.ndarray`, optional): Numerical part of the dataset.
                Can be null if a categorical part is present.
                The objects are encoded row-wise, and the features column-wise.
                The number of rows of ``X`` must match those of ``Xc``, 
                if present. Must be of ``dtype=float64`` and ``f_contiguous``.
            Xc (:obj:`numpy.ndarray`, optional): Categorical part of the 
                dataset. Can be null if a numerical part is present.
                The objects are encoded row-wise, and the features column-wise.
                The number of rows of ``Xc`` must match those of ``X``,
                if present. Must be of ``dtype=int32`` and ``f_contiguous``.
            y (:obj:`numpy.ndarray`, optional): Labels of the dataset.
                Must be of ``dtype=int32`` and ``f_contiguous``.
        
        Raises:
            ValueError: If both ``X`` and ``Xc`` are empty, or if one of the
                arguments has the wrong format, as specified in their argument
                documentation.
        """
        self.X = X
        self.Xc = Xc
        self.y = y
        if X is None and Xc is None:
            raise ValueError("At least one of the numerical and categorical"
                             " parts of a dataset must be non-empty.")
        if X is not None:
            if type(X) != np.ndarray:
                raise ValueError("The numerical part of a dataset must be a "
                                 "numpy.ndarray.")
            if X.dtype != "float64":
                raise ValueError("The numerical part of a dataset must be of "
                                 "dtype float64.")
            if not X.flags.f_contiguous:
                raise ValueError("The numerical part of a dataset must be such"
                                 " that flags.f_contiguous is True.")
        if Xc is not None:
            if type(Xc) != np.ndarray:
                raise ValueError("The categorical part of a dataset must be a "
                                 "numpy.ndarray.")
            if Xc.dtype != "int32":
                raise ValueError("The categorical part of a dataset must be "
                                 "of dtype int32.")
            if not Xc.flags.f_contiguous:
                raise ValueError("The categorical part of a dataset must be "
                                 "such that flags.f_contiguous is True.")
        if y is not None:
            if type(y) != np.ndarray:
                raise ValueError("The labels part of a dataset must be a "
                                 "numpy.ndarray.")
            if y.dtype != "int32":
                raise ValueError("The labels part of a dataset must be of "
                                 "dtype int32.")
            if not y.flags.f_contiguous:
                raise ValueError("The labels part of a dataset must be such "
                                 "that flags.f_contiguous is True.")
        if X is not None and Xc is not None and X.shape[0] != Xc.shape[0]:
            raise ValueError("The number of rows of the numerical and "
                             "categorical part of a dataset must match.")
        if y is not None and y.shape[0] != self.size():
            raise ValueError("The number of labels in a dataset must match the"
                             " size of the dataset")
        if self.X is not None and len(self.X.shape) != 2:
            raise ValueError("Invalid shape for numerical features " +
                             str(self.X.shape) + " vs (n, m)")
        if self.Xc is not None and len(self.Xc.shape) != 2:
            raise ValueError("Invalid shape for categorical features " +
                             str(self.Xc.shape) + " vs (n, m)")
        self.Xnames = (["" for x in range(self.X.shape[1])] 
                       if self.X is not None else [])
        self.Xcnames = (["" for x in range(self.Xc.shape[1])]
                        if self.Xc is not None else [])
        self.Xnamedict = {}
        self.Xcnamedict = {}
        self.yname = ""

    def set_real_names(self, names):
        """Sets the names of the real features.

        Args:
            names (list[str] or dict[str, int]): A dictionary mapping a feature
                name to its column index in self.X, or a collection indexed by
                integers as long as the number of columns of self.X.

        Raises:
            ValueError: If names is of the wrong type.
            RuntimeError: If one of the indices is larger than the number of
                columns of self.X, or, if names is a list, if it is smaller
                than the number of columns of X. It is not guaranteed that the
                internal state of the object has not changed.
        """
        if self.X is None:
            return
        if isinstance(names, dict):
            self.Xnamedict = {}
            self.Xnames = ["" for x in range(self.X.shape[1])]
            for k in names:
                if not isinstance(names[k], int):
                    raise RuntimeError(
                          "the range of names must contain only "
                          "integers, found " + str(type(names[k]))
                    )
                if names[k] < self.X.shape[1]:
                    self.Xnamedict[k] = names[k]
                    self.Xnames[names[k]] = k
                else:
                    raise RuntimeError(
                          "index %d is larger than the number of "
                          "features of X (%d)" % (names[k], self.X.shape[1])
                    )
        else:
            self.Xnamedict = {}
            self.Xnames = ["" for x in range(self.X.shape[1])]
            for i in range(len(names)):
                if not isinstance(names[i], str):
                    raise RuntimeError(
                          "the list of names must contain "
                          "only strings, found " + str(type(names[i]))
                    )
                self.Xnames[i] = names[i]
                self.Xnamedict[names[i]] = i

    def set_categorical_names(self, names):
        """Sets the names of the categorical features.

        Args:
            names (list[str] or dict[str, int]): A dictionary mapping a feature
                name to its column index in self.Xc, or a collection indexed
                by integers as long as the number of columns of self.Xc.

        Raises:
            RuntimeError: If one of the indices is larger than the number of
                columns of self.Xc, or, if names is a list, if it is smaller
                than the number of columns of Xc. It is not guaranteed that
                the internal state of the object has not changed.
        """
        if self.Xc is None:
            return
        if isinstance(names, dict):
            self.Xcnamedict = {}
            self.Xcnames = ["" for x in range(self.Xc.shape[1])]
            for k in names:
                if not isinstance(names[k], int):
                    raise RuntimeError(
                          "the range of names must contain only "
                          "integers, found " + str(type(names[k]))
                    )
                if names[k] < self.Xc.shape[1]:
                    self.Xcnamedict[k] = names[k]
                    self.Xcnames[names[k]] = k
                else:
                    raise RuntimeError(
                          "index %d is larger than the number of "
                          "features of X (%d)" % (names[k], self.Xc.shape[1])
                    )
        else:
            self.Xcnamedict = {}
            self.Xcnames = ["" for x in range(self.Xc.shape[1])]
            for i in range(len(names)):
                if not isinstance(names[i], str):
                    raise RuntimeError(
                          "the list of names must contain "
                          "only strings, found " + str(type(names[i]))
                    )
                self.Xcnames[i] = names[i]
                self.Xcnamedict[names[i]] = i

    def set_labels_name(self, name):
        """Sets the name of the label.

        Args:
            name (:obj:`str`, optional): Name of the label. 
                If None, an empty string is inserted instead.
        
        Raises:
            TypeError: If name is not a string.
        """
        if not isinstance(name, str) and name is not None:
            raise TypeError("The label name must be a string.")
        self.yname = name if name is not None else ""

    def size(self):
        """Retrieves the size of the dataset.

        Returns:
            int: The size of the dataset.
        """
        return self.X.shape[0] if self.X is not None else self.Xc.shape[0]

    def get_feature_count(self):
        """Returns the number of features.

        Returns:
            int: The number of features the data has.
        """
        real_count = self.X.shape[1] if self.X is not None else 0
        cat_count = self.Xc.shape[1] if self.Xc is not None else 0
        return real_count + cat_count

    def get_feature_name(self, i):
        """Converts a feature id to its name.

        Args:
            i (int): The index of the feature.
        
        Returns:
            str: The name of the feature.
        """
        if self.X is not None and i < self.X.shape[1]:
            return "att" + str(i) if self.Xnames[i] == "" else self.Xnames[i]
        else:
            i = i - (self.X.shape[1] if self.X is not None else 0)
            return "att" + str(i) if self.Xcnames[i] == "" else self.Xcnames[i]

    def feature_set_to_str(self, fs):
        """Converts a feature set to string.

        Args:
            fs(list[int]): A feature set represented as a list of ids.
        
        Returns:
            str: A string representation of the input feature set.
        """
        if len(fs) >= self.get_feature_count():
            return "full"
        if len(fs) == 0:
            return "{}"
        return "{ " + ", ".join([self.get_feature_name(i) for i in fs]) + " }"

    def split(self, splitter):
        """Generates train-test splits.

        Args:
            splitter: Object implementing  a method ``split(X, y)`` or 
                ``split(X)``, to which the generation of the index-sets
                for the train-test splits is relayed.

        Yields:
            A pair of :obj:`mlconcepts.data.Dataset` objects representing a
            train/test split. 
            
            More specifically, a generator is returned by the splitter
            ``split`` method. This generator yields indices sets for the train
            and test set at each iteration. These indices sets are used to
            sample data for the construction of the yielded Dataset objects. 
        """
        #Creates the generator from the splitter. If y is null, it calls 
        #split(X), otherwise split(X, y). Also checks whether one of X and Xc
        #is null, and picks one of the two to generate the index sets.
        #Both the datasets will be splitted in the output.
        generator = None
        if self.y is not None:
            generator = (splitter.split(self.Xc, self.y) if self.X is None else
                         splitter.split(self.X, self.y))
        else:
            generator = (splitter.split(self.Xc) if self.X is None else
                         splitter.split(self.X))
        for train_indices, test_indices in generator:
            #Split the matrices
            x_train  = (None if self.X is None else
                        np.asfortranarray(self.X[train_indices]))
            xc_train = (None if self.Xc is None else
                        np.asfortranarray(self.Xc[train_indices]))
            y_train  = (None if self.y is None else
                        np.asfortranarray(self.y[train_indices]))
            x_test  = (None if self.X is None else
                       np.asfortranarray(self.X[test_indices]))
            xc_test = (None if self.Xc is None else
                       np.asfortranarray(self.Xc[test_indices]))
            y_test  = (None if self.y is None else
                       np.asfortranarray(self.y[test_indices]))
            #And create the datasets
            train = Dataset(X = x_train, Xc = xc_train, y = y_train)
            test = Dataset(X = x_test, Xc = xc_test, y = y_test)
            #Transfer the names to the splits
            train.set_real_names(self.Xnames)
            test.set_real_names(self.Xnames)
            train.set_categorical_names(self.Xcnames)
            test.set_categorical_names(self.Xcnames)
            train.set_labels_name(self.yname)
            test.set_labels_name(self.yname)
            yield train, test

def basic_load(dataset, categorical=[], labels=None, Xc=None, y=None,
               settings={}):
    """Dummy dataset loader.
    
    This function should not be called directly, but rather indirectly via
    :func:`mlconcepts.data.load`.

    Args:
        dataset (:obj:`mlconcepts.data.Dataset`): A dataset.
        categorical: Ignored.
        labels: Ignored.
        Xc: Ignored.
        y: Ignored.
        settings: Ignored.

    Returns:
        :obj:`mlconcepts.data.Dataset`: The dataset.
    """
    return dataset