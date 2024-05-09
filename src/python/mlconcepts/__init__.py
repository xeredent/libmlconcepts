"""Implements several machine learning algorithms based on FCA.

Currently, this version of the library only implements the outlier detection
algorithms introduced
`here <https://www.sciencedirect.com/science/article/pii/S0167923624000290>`_.
"""

from .SODModel import SODModel # noqa: F401
from .UODModel import UODModel # noqa: F401
from mlconcepts.data import load # noqa: F401