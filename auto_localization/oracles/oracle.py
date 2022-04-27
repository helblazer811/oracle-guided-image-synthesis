import numpy as np

"""
    This is an interface for Oracle classes.
    The core functionality of an Oracle is being able to 
    answer queries that are given to it. 
"""
class IOracle():

    """
        Takes in a query and returns an 
        answer to that query

        query = ((anchor, item_a, item_b), index)
        
        item_a and item_b are numpy arrays that represent images

        The reason for this structure is because some Oracles
        use the index to answer and some use the data
    """
    def answer_query(self, query, single_feature_triplet=False):
        raise NotImplementedError


"""
    This is a class that impliments the IOracle interface
    by taking several other Oracles and ensambling their answers
    through a defined weighting matrix. 
"""
class EnsembleOracle(IOracle):

    def __init__(self, oracles, weighting):
        if len(oracles) != len(weighting):
            raise Exception
        self.oracles = oracles
        self.weighting = np.array(weighting)
    
    """
        Gets an answer from each query and ensambles them
    """
    def answer_query(self, query, single_feature_triplet=False):
        # get answers
        answers = [oracle.answer_query(query) for oracle in self.oracles]
        # convert to onehot
        onehot_answers = np.eye(2)[answers]
        # weighted sum
        weighted_answers = self.weighting[None, :].T * onehot_answers
        summed_answers = np.sum(weighted_answers, axis=1)
        # final answer
        return np.argmax(summed_answers)
