import pandas
import numpy
from .data import graph_load_elliptic

class TestGraphDatasetLoad: # noqa: D101

    def test_load_elliptic(self): # noqa: D102
        features = pandas.DataFrame({
            "txId" : [563, 641, 353],
            "feat" : [.1, .2, .3]
        })
        edges = pandas.DataFrame({
            "txId1" : [563, 563, 641, 353, 353],
            "txId2" : [563, 641, 353, 641, 563]
        })
        classes = pandas.DataFrame({
            "txId" : [641, 563, 353],
            "class" : [0, 1, 1]
        })
        g = graph_load_elliptic(
            features=features,
            edges=edges,
            classes=classes,
            src_name="txId1",
            dst_name="txId2",
            features_tx_name="txId",
            classes_tx_name="txId",
            class_name="class"
        )
        assert (g.sources == numpy.array([1, 1, 2, 0, 0])).all()
        assert (g.targets == numpy.array([1, 2, 0, 2, 1])).all()
        assert (g.y == numpy.array([1, 1, 0])).all()
        assert (g.nodes.X == numpy.array([[.3], [.1], [.2]])).all()


    def test_load_elliptic_no_id(self): # noqa: D102
        features = pandas.DataFrame({
            "feat" : [.1, .2, .3]
        })
        edges = pandas.DataFrame({
            "txId1" : [0, 0, 1, 2, 2],
            "txId2" : [0, 1, 1, 1, 2]
        })
        classes = pandas.DataFrame({
            "class" : [0, 1, 1]
        })
        g = graph_load_elliptic(
            features=features,
            edges=edges,
            classes=classes,
            src_name="txId1",
            dst_name="txId2",
            class_name="class"
        )
        assert (g.sources == numpy.array([0, 0, 1, 2, 2])).all()
        assert (g.targets == numpy.array([0, 1, 1, 1, 2])).all()
        assert (g.y == numpy.array([0, 1, 1])).all()
        assert (g.nodes.X == numpy.array([[.1], [.2], [.3]])).all()

    def test_load_elliptic_overlap(self): # noqa: D102
        features = pandas.DataFrame({
            "txId" : [563, 641, 353],
            "feat" : [.1, .2, .3],
            "class" : [0, 1, 1]
        })
        edges = pandas.DataFrame({
            "txId1" : [563, 563, 641, 353, 353],
            "txId2" : [563, 641, 353, 641, 563]
        })
        g = graph_load_elliptic(
            features=features,
            edges=edges,
            classes=features,
            src_name="txId1",
            dst_name="txId2",
            features_tx_name="txId",
            classes_tx_name="txId",
            class_name="class"
        )
        assert (g.sources == numpy.array([1, 1, 2, 0, 0])).all()
        assert (g.targets == numpy.array([1, 2, 0, 2, 1])).all()
        assert (g.y == numpy.array([1, 0, 1])).all()
        assert (g.nodes.X == numpy.array([[.3], [.1], [.2]])).all()

