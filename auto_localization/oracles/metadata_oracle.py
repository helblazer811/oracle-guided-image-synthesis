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
class MetadataOracle(oracle.IOracle):
    
    """
        Takes in metadata dataset and image dataset of a certain format
    """
    def __init__(self, metadata_dataset=None, component_weighting=None):
        # the dataset is assumed to be a predefined triplet dataset
        self.metadata_dataset = metadata_dataset
        self.component_weighting = np.array(component_weighting)
        
    """
        Calculate metadata based on image
    """
    def calculate_metadata(self, image):
        if len(np.shape(image)) > 2:
            image = image.squeeze()
        return self.metadata_dataset.measure_image(image)

    def calculate_weighted_metadata(self, image):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        metadata = self.calculate_metadata(image)
        return self.component_weighting * metadata
        #return np.dot(metadata, self.component_weighting.T)

    def calculate_metadata_distance(self, a, b):
        metadata_a = self.calculate_metadata(a)
        metadata_b = self.calculate_metadata(b)
        # weighted metadatas
        weighted_a = self.component_weighting * metadata_a
        weighted_b = self.component_weighting * metadata_b
        # distance
        distance = np.linalg.norm(weighted_a - weighted_b)
        return distance 

    def answer_query_from_metadata(self, query, single_feature_triplet=False):
        anchor, a, b = query
        # weighted metadatas
        weighted_anchor = self.component_weighting * anchor
        weighted_a = self.component_weighting * a
        weighted_b = self.component_weighting * b
        # distance
        distance_a = np.linalg.norm(weighted_a - weighted_anchor)
        distance_b = np.linalg.norm(weighted_b - weighted_anchor)
        # get answer
        answer = 0 if distance_a < distance_b else 1
        return answer
   
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
        distance_a = self.calculate_metadata_distance(reference, item_a)
        distance_b = self.calculate_metadata_distance(reference, item_b)
        # get answer
        answer = 0 if distance_a < distance_b else 1
        return answer

