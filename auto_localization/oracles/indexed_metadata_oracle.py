import sys
sys.path.append("../..")
import auto_localization.oracles.oracle as oracle
import numpy as np
import torch

"""
    This oracle answers queries based on metadata, but
    instead of accepting images it indexes to a given dataset. 
    This helps during training when getting metadata is 
    expensive. This is kindof inbetween PredefinedOracle
    and MetadataOracle

    TODO handle categorical metadata features
"""
class IndexedMetadataOracle(oracle.IOracle):
    
    """
        Takes in metadata dataset and image dataset of a certain format
    """
    def __init__(self, metadata_dataset=None, component_weighting=None, inject_triplet_noise=0.0):
        self.metadata_dataset = metadata_dataset
        if component_weighting is None:
            component_weighting = np.ones(np.shape(self.metadata_dataset[0])[-1])
        # the dataset is assumed to be a predefined triplet dataset
        self.inject_triplet_noise = inject_triplet_noise
        self.component_weighting = np.array(component_weighting)
 
    def calculate_weighted_metadata(self, index):
        metadata = self.metadata_dataset[index]
        return np.dot(metadata, self.component_weighting.T)

    def calculate_metadata_distance(self, a, b, attribute_index=None):
        component_weighting = self.component_weighting
        if not attribute_index is None:
            component_weighting = np.zeros_like(self.component_weighting)
            component_weighting[attribute_index] = 1
        metadata_a = self.metadata_dataset[a]
        metadata_b = self.metadata_dataset[b]
        if not self.inject_triplet_noise == 0.0:
            # inject random noise into slant and thickness
            attribute_indices = torch.nonzero(torch.Tensor(self.component_weighting)).view(-1)
            for attribute_index in attribute_indices:
                noise_amount = np.random.normal(loc=0.0, scale=self.inject_triplet_noise)
                metadata_a[attribute_index] += noise_amount
                noise_amount = np.random.normal(loc=0.0, scale=self.inject_triplet_noise)
                metadata_b[attribute_index] += noise_amount


        distance = np.dot(np.abs(metadata_a - metadata_b), component_weighting.T)

        return distance 
    
    def answer_query_single_feature(self, query, attribute_index=None):
        # unpack query
        reference_index, a_index, b_index = query
        # distance between two 
        distance_a = self.calculate_metadata_distance(a_index, reference_index, attribute_index=attribute_index)
        distance_b = self.calculate_metadata_distance(b_index, reference_index, attribute_index=attribute_index)
        # get answer
        # perform elementwise less than 
        answers = np.logical_not(np.less(distance_a, distance_b))
        # answer = 0 if np.less(distance_a, distance_b) else 1
        return answers
       
    """
        Returns an answer to a query based on 
        pre-defined query answers.

        queries come in the form of 
        query = (reference, item_a, item_b)
    """
    def answer_query(self, query, single_feature_triplet=False, attribute_index=None, inject_triplet_noise=0.0):
        if single_feature_triplet:
            return self.answer_query_single_feature(query, attribute_index=attribute_index)
        # unpack query
        reference_index, a_index, b_index = query
        # distance between two 
        distance_a = self.calculate_metadata_distance(a_index, reference_index)
        distance_b = self.calculate_metadata_distance(b_index, reference_index)
        # get answer
        # perform elementwise less than 
        answers = np.logical_not(np.less(distance_a, distance_b))
        # answer = 0 if np.less(distance_a, distance_b) else 1
        return answers

    """
        Answers a batch of queries
    """
    def answer_query_batch(self, queries):
        pass
