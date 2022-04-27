import sys
sys.path.append("../..")
import os
import auto_localization.oracles.oracle as oracle
from auto_localization.oracles.pairwise_distance_matrix import PairwiseDistanceMatrix
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

"""
    This is an oracle made out of a distance matrix
"""
class MatrixOracle(oracle.IOracle):
    
    """
        Takes in metadata dataset and image dataset of a certain format
    """
    def __init__(self, path=None):
        # load up a pretrained resnet model that works on 32x32 images
        self.path = path
        self.distance_matrix = self.load_distance_matrix(self.path)
        self.indexed = True

    def load_distance_matrix(self, path):
        distance_matrix = PairwiseDistanceMatrix()
        distance_matrix.unserialize(path)
        return distance_matrix

    def calculate_distance(self, index_a, index_b, attribute_index):
        return self.distance_matrix.calculate_distance(index_a, index_b, attribute_index)

    """
        Return a triplet
    """
    def generate_triplet(self, single_feature_triplet=False, attribute_index=-1):
        # select three indices in the range of the indices
        num_indices = len(self.distance_matrix.indices)
        index = np.random.choice(num_indices, size=1)[0]
        index_a = np.array(index).astype(np.int_)
        while index_a == index:
            index_a = np.random.choice(np.arange(0, num_indices)).astype(np.int_)
            index_b = index
        while index_b == index or index_b == index_a:
            index_b = np.random.choice(np.arange(0, num_indices)).astype(np.int_)
        # get ordering of the indices
        anchor = index
        distance_a = self.calculate_distance(anchor, index_a, attribute_index=attribute_index)
        distance_b = self.calculate_distance(anchor, index_b, attribute_index=attribute_index)
        # get the correct ordering
        answer = 0 if distance_a < distance_b else 1
        if answer == 0:
            positive = index_a
            negative = index_b
        else:
            positive = index_b
            negative = index_a
        # map the indices to their image dataset indices
        anchor = int(self.distance_matrix.indices[anchor].item())
        positive = int(self.distance_matrix.indices[positive].item())
        negative = int(self.distance_matrix.indices[negative].item())
        return anchor, positive, negative, attribute_index
    
    """
        Returns an answer to a query based on 
        pre-defined query answers.

        queries come in the form of 
        query = (reference, item_a, item_b)
    """
    def answer_query(self, query, **kwargs):
        # unpack query
        reference, item_a, item_b = query
        single_feature_triplet = kwargs["single_feature_triplet"] if "single_feature_triplet" in kwargs else False
        attribute_index = kwargs["attribute_index"] if "attribute_index" in kwargs else -1
        # distance between two 
        distance_a = self.calculate_distance(reference, item_a, attribute_index=attribute_index)
        distance_b = self.calculate_distance(reference, item_b, attribute_index=attribute_index)
        # get answer
        answer = 0 if distance_a < distance_b else 1
        return answer

