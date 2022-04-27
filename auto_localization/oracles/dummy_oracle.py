import sys
sys.path.append("../..")
import auto_localization.oracles.oracle as oracle
import numpy as np
import torch

"""
    This is an oracle that answers queries based on 
    some way of calculating metadata. Each of the metadata
    is assumed to be continuous and normalized.

    TODO handle categorical metadata features
"""
class DummyOracle(oracle.IOracle):
    
    """
        Takes in metadata dataset and image dataset of a certain format
    """
    def __init__(self, component_weighting=None):
        # the dataset is assumed to be a predefined triplet dataset
        self.component_weighting = component_weighting
  
    """
        Returns an answer to a query based on 
        pre-defined query answers.

        queries come in the form of 
        query = (reference, item_a, item_b)
    """
    def answer_query(self, query, single_feature_triplet=False):
        # unpack query
        reference, item_a, item_b = query
        # distance between two 
        weghted_reference = self.component_weighting * reference
        weighted_a = self.component_weighting * item_a
        weighted_b = self.component_weighting * item_b
        # distance
        distance_a = np.linalg.norm(weighted_a - weighted_reference)
        distance_b = np.linalg.norm(weighted_b - weighted_reference)
        # get answer
        answer = 0 if distance_a < distance_b else 1
        return answer

