import numpy

from .SODModel import SODModel
from .UODModel import UODModel

class TestModels:
    X = numpy.array(
        [
            [5.3, 3.2, 2.4],
            [3.1, 8.7, 4.2],
            [3.2, 8.8, 5.2],
            [1.2, 5.3, 8.8],
            [3.2, 1.2, 1.0],
            [9.0, 8.9, 2.0],
            [3.6, 1.7, 1.4],
            [5.6, 6.9, 2.3],
        ],
        order = "F",
        dtype = numpy.float64
    )
    Xc = numpy.array(
        [
            [0],
            [1],
            [1],
            [0],
            [0],
            [1],
            [1],
            [0]
        ],
        order = "F",
        dtype = numpy.float64
    )
    y = numpy.array(
        [0, 1, 1, 0, 0, 1, 0, 0],
        order = "F",
        dtype = numpy.int32
    )
    tX = numpy.array([[2.2, 8.75, 6.3], [5.4, 2.2, 1.7]], 
                     order = "F", dtype =numpy.float64)
    tXc = numpy.array([[1], [0]], order = "F", dtype = numpy.int32)
    ty = numpy.array([[1], [0]], order = "F", dtype = numpy.int32)

    def test_sup_explanation_predictions(self):
        model = SODModel(n = 4)
        model.fit(self.X, Xc = self.Xc, y = self.y)
        pred = model.predict(self.tX, Xc = self.tXc, y = self.ty)
        expl = model.predict_explain(self.tX, Xc = self.tXc, y = self.ty)
        assert len(pred) == len(expl.get_predictions())
        for i in range(len(pred)):
            assert abs(pred[i] - expl.get_predictions()[i]) < 0.00001

    def test_unsup_explanation_predictions(self):
        model = UODModel(n = 4)
        model.fit(self.X, Xc = self.Xc, y = self.y)
        pred = model.predict(self.tX, Xc = self.tXc, y = self.ty)
        expl = model.predict_explain(self.tX, Xc = self.tXc, y = self.ty)
        assert len(pred) == len(expl.get_predictions())
        for i in range(len(pred)):
            assert abs(pred[i] - expl.get_predictions()[i]) < 0.00001