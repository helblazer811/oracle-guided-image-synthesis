import sys
sys.path.append("../..")
import auto_localization.oracles.oracle as oracle
import numpy as np
import torch

"""
    This oracle basically compares the classes of data.
    Very simple. 
    TODO handle categorical metadata features
"""
class IndexedClassOracle(oracle.IOracle):
    
    """
        Takes in image dataset of a certain format
    """
    def __init__(self, image_dataset=None):
        # the dataset is assumed to be a predefined triplet dataset
        self.image_dataset = image_dataset
        
    """
        Answers a query of the form

        (anchor, item_a, item_b)
    """
    def answer_query(self, query):
        anchor_index, index_a, index_b = query
        # get identity matrix of size of labels
        num_labels = torch.max(self.image_dataset.labels)
        label_identity = np.eye(num_labels+1)
        # get classes 
        anchor_class = label_identity[self.image_dataset.labels[anchor_index]]
        a_class = label_identity[self.image_dataset.labels[index_a]]
        b_class = label_identity[self.image_dataset.labels[index_b]]
        # distances
        a_diff = np.linalg.norm(anchor_class - a_class)
        b_diff = np.linalg.norm(anchor_class - b_class)
        # chose the correct answer
        closer = 0 if a_diff < b_diff else 1
        return closer




