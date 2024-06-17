"""This module exposes classes that represent explanation data."""

import numpy


class ExplanationEntry(object):
    """Represents the explanation for an object in a dataset.

    This is essentially a wrapper around a dictionary with some additional
    utility functions.
    """

    def __init__(self, expl_data = {}, dataset = None, prediction = 0.0):
        """Constructs an explanation entry.

        Args:
            expl_data (list[(str,float)]): A list of pairs mapping a string
                representing a feature set to its relevance.
            dataset (:class:`mlconcepts.data.Dataset`): A dataset object which
                holds the data for which the prediction has been made.
            prediction (float): The prediction for the object this entry is
                related to.
        """
        self.expl_data = expl_data
        self.expl_data.sort(key=lambda it: it[1], reverse=True)
        self.dataset = dataset
        self.prediction = prediction

    def __iter__(self):
        """Iterates through the entry.

        Returns:
            A generator producing pairs of type (str, float) containing the
            string representation of a feature set, and its relevance.
        """
        for k, v in self.expl_data:
            yield k, v

    def __str__(self):
        """Converts the explanation entry to a string.
        
        Returns:
            str: A string representation of this explanation entry.
        """
        best = [ self.expl_data[i][0] + " : " + str(self.expl_data[i][1])
                 for i in range(min(3, len(self.expl_data))) ]
        if len(best) == 0:
            return "{}"
        if len(best) > self.dataset.get_feature_count():
            best.append("...")
        return ("Prediction: " + str(self.prediction) + ". Explainers: " + 
               "{ " + ", ".join(best) + " }")

class ExplanationData(object):
    """Represents explanation data together with predictions.

    ExplanationData objects also have facilities that convert and access
    explanation data based on feature names.
    """

    def __init__(self, predictions = numpy.array([]), 
                 outdegs = numpy.array([]), feature_sets = [], dataset = None):
        """Constructs an ExplanationData object.

        Args:
            predictions (numpy.array): An array containing the predictions
                made by a machine learning model.
            outdegs (numpy.array): A matrix containing for each agenda
                (row-wise) and for each object (column-wise) a score
                representing how much the contribution of the agenda (feature
                set) matters in the final prediction.
            feature_sets (list[list[int]]): A list of feature sets. Each
                feature set is a list of integers, each representing a feature
                in the set.
            dataset (:class:`mlconcepts.data.Dataset`): A dataset object which
                holds the data for which the predictions have been made.
        """
        self.predictions = predictions
        self.outdegs = outdegs
        self.feature_sets = feature_sets
        self.dataset = dataset

    def get_predictions(self):
        """Returns the predictions vector.

        Returns:
            numpy.array: A vector containing a prediction value for each 
                object.
        """
        return self.predictions

    def get_explanation(self, id):
        """Returns the explanation data for some object.

        Returns:
            :class:`mlconcepts.ExplanationEntry`: An object representing data
            regarding an interpretable predictions for the object.
        """
        return ExplanationEntry(
            dataset = self.dataset,
            prediction = self.predictions[id],
            expl_data = [
                (
                    self.dataset.feature_set_to_str(self.feature_sets[i]), 
                    self.outdegs[i][id]
                ) for i in range(len(self.feature_sets))
            ]
        )
    
    def __getitem__(self, key):
        """Returns the explanation data for some object.

        Returns:
            :class:`mlconcepts.ExplanationEntry`: An object representing data
            regarding an interpretable predictions for the object.
        """
        return self.get_explanation(key)
    
