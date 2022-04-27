import sys
sys.path.append("../..")
import auto_localization.oracles.oracle as oracle

"""
    This is an oracle that answers queries based on 
    pre-defined triplet queries. It takes in a pre-existing
    metadata dataset.
"""
class PredefinedOracle(oracle.IOracle):
    
    """
        Takes in triplet datasets of a certain format
    """
    def __init__(self, triplet_dataset, image_dataset):
        # the dataset is assumed to be a predefined triplet dataset
        self.triplet_dataset = triplet_dataset
        self.image_dataset = image_dataset
        self.reference_index = None
        self.reference_queries = None
        
    """
        Sets the reference for a specific usage of the 
        Oracle
    """
    def set_reference(self, reference_index):
        self.reference_index = reference_index

    """
        Gets queries corresponding to this reference 
    """
    def get_reference_queries(self):
        if self.reference_queries != None:
            return self.reference_queries
        reference_queries = []
        for i in range(self.triplet_dataset.__len__()):
            triplet = self.triplet_dataset.__getitem__(i)
            if triplet[0] == self.reference_index:
                reference_queries.append(triplet)

        self.reference_queries = reference_queries

    """
        Returns an answer to a query based on 
        pre-defined query answers.

        queries come in the form of 
        query = ((anchor, item_a, item_b), index)
    """
    def answer_query(self, query):
        #if self.reference 
        triplet, index = query
        anchor, item_a, item_b = triplet
        # get answer at index
        answer_index = self.triplet_dataset.triplet_indices[index][0]
        answer = self.image_dataset.images[answer_index] 
        # chose which answer 
        return answer

