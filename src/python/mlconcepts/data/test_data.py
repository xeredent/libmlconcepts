# noqa: D100
import pandas
import numpy

from .PandasLoader import pandas_load
from .NumpyLoader import numpy_load

class TestPandas(object): # noqa: D101
    df = pandas.DataFrame({
        "diameter" : [
            3.5,
            4.5,
            5.3,
            6.5,
            2.1
        ],
        "color" : [
            "green",
            "red",
            "green",
            "green",
            "yellow"
        ],
        "delicious" : [
            "yes",
            "yes",
            "yes",
            "yes",
            "no"
        ]
    })

    def test_pandas_generic_load(self): # noqa: D102
        d = pandas_load(self.df, labels = "delicious") # noqa: F841
        assert d.X.shape == (5, 1)
        assert d.Xc.shape == (5, 1)
        assert str(d.y) == "[0 0 0 0 1]"
        assert d.X.flags.f_contiguous
        assert d.Xc.flags.f_contiguous
        assert d.X.dtype == numpy.float64
        assert d.Xc.dtype == numpy.int32
        assert d.y.dtype == numpy.int32
        assert str(d.Xnames) == "['diameter']"
        assert str(d.Xcnames) == "['color']"
        assert d.yname == "delicious"

    def test_pandas_override_labels(self): # noqa: D102
        new_labels = numpy.array(
            [1, 0, 1, 1, 0], 
            order = "F", 
            dtype = numpy.int32
        )
        d = pandas_load(self.df, labels = "delicious", y = new_labels)
        assert d.X.shape == (5, 1)
        assert d.Xc.shape == (5, 1)
        assert str(d.y) == "[1 0 1 1 0]"
        assert d.X.flags.f_contiguous
        assert d.Xc.flags.f_contiguous
        assert d.X.dtype == numpy.float64
        assert d.Xc.dtype == numpy.int32
        assert d.y.dtype == numpy.int32
        assert str(d.Xnames) == "['diameter']"
        assert str(d.Xcnames) == "['color']"
        assert d.yname == "delicious"

    def test_pandas_categorical_integer(self): # noqa: D102
        pdf = pandas.DataFrame({ "feat" : [1, 1, 1, 1, 0] })
        d = pandas_load(pdf, categorical = ["feat"])
        assert d.X is None
        assert d.Xc.shape == (5, 1)
        assert d.y is None
        assert d.Xc.flags.f_contiguous
        assert d.Xc.dtype == numpy.int32
        assert str(d.Xcnames) == "['feat']"

class NumpyTest(object): # noqa: D101

    def test_load_wrong_types(self): # noqa: D102
        X = numpy.array(
            [
                [2.0, 3.25, 2.4],
                [9.5, 1.33, 1.22]
            ],
            order = "C",
            dtype = numpy.float32
        )
        Xc = numpy.array(
            [ [1], [0] ],
            order = "C",
            dtype = numpy.int8
        )
        d = numpy_load(X, Xc = Xc)
        assert X.flags.f_contiguous
        assert X.dtype == numpy.float64
        assert Xc.flags.f_contiguous
        assert Xc.dtype == numpy.int32
        assert X.shape == (2, 3)
        assert Xc.shape == (2, 1)

class SplitGen(object):
    def split(self, X, y): # noqa: D103
        yield [1, 2], [0]
        yield [0, 2], [1]
        yield [0], [1, 2]

class TestDataset: # noqa: D101

    def test_datasplit(self): # noqa: D102
        X = numpy.array([ [2.1], [2.6], [3.1] ])
        Xc = numpy.array([ [2], [0], [1] ])
        y = numpy.array([ [0], [1], [0] ])
        dataset = numpy_load(X, Xc = Xc, y = y)
        gen = dataset.split(SplitGen())
        train, test = next(gen)
        assert train.X.shape == (2, 1)
        assert train.X[0][0] == 2.6
        assert train.X[1][0] == 3.1
        assert train.Xc[0][0] == 0
        assert train.Xc[1][0] == 1
        assert train.y[0] == 1
        assert train.y[1] == 0
        assert test.X.shape == (1, 1)
        assert test.X[0][0] == 2.1
        assert test.Xc[0][0] == 2
        assert test.y[0] == 0
        train, test = next(gen)
        assert train.X.shape == (2, 1)
        assert train.X[0][0] == 2.1
        assert train.X[1][0] == 3.1
        assert train.Xc[0][0] == 2
        assert train.Xc[1][0] == 1
        assert train.y[0] == 0
        assert train.y[1] == 0
        assert test.X.shape == (1, 1)
        assert test.X[0][0] == 2.6
        assert test.Xc[0][0] == 0
        assert test.y[0] == 1
        train, test = next(gen)
        assert train.X.shape == (1, 1)
        assert train.X[0][0] == 2.1
        assert train.Xc[0][0] == 2
        assert train.y[0] == 0
        assert test.X.shape == (2, 1)
        assert test.X[0][0] == 2.6
        assert test.X[1][0] == 3.1
        assert test.Xc[0][0] == 0
        assert test.Xc[1][0] == 1
        assert test.y[0] == 1
        assert test.y[1] == 0



