import sys
sys.path.append("../..")
import auto_localization.oracles.oracle as oracle

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
    def __init__(self, metadata_function=None, component_weighting=None):
        # the dataset is assumed to be a predefined triplet dataset
        self.metadata_function = metadata_function
        self.reference_index = None
        self.reference_queries = None
        self.component_weighting = component_weighting 
        
    """
        Sets the reference for a specific usage of the 
        Oracle
    """
    def set_reference(self, reference_image):
        self.reference_image = reference_image
        self.reference_metadata = self.calculate_metadata(self.reference_image)
    
    """
        Calculate metadata based on image
    """
    def calculate_metadata(self, image):
        # TODO implement morpho mnist metadata calculator
       return self.metadata_function(image)

    """
        Returns an answer to a query based on 
        pre-defined query answers.

        queries come in the form of 
        query = (item_a, item_b)
    """
    def answer_query(self, query):
        # unpack query
        item_a, item_b = query
        # calculate metadata distances
        metadata_a = self.calculate_metadata(item_a)
        metadata_b = self.calculate_metadata(item_b)
        # distance between two 
        distance_a = np.abs(metadata_a - self.reference_metadata, axis=1) * self.component_weighting
        distance_b = np.abs(metadata_b - self.reference_metadata, axis=1) * self.component_weighting
        # chose which answer 
        answer = 0 if distance_a < distance_b else 1
        return answer

