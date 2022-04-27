from localizers.approximate_mcmv_localization import ApproximateMCMVLocalizer
from random import randrange

"""
    This is a wrapper for a localization experiment 
    that uses a PredefinedOracle
"""
class PredefinedRollout():

    def __init__(self, model=None, oracle=None, data_manager=None, method="mcmv"):
        self.model = model
        self.oracle = oracle
        self.data_manager = data_manager
        self.method = method
        self.reference_index = None
        self.reference_data = None
        # get localizer based on method
        if self.method == "mcmv":
            latent_dim = self.model.z_dim
            self.localizer = ApproximateMCMVLocalizer(ndim=latent_dim)
        else:
            raise Exception
        # setup 
        self._setup_reference_data()
        self._generate_query_pairs()
    
    """
        Get a random reference value and get all of the corresponding 
        query triplets for it
    """
    def _setup_reference_data(self, train=False):
        # get the triplet dataset
        if train:
            triplet_dataset = self.data_manager.triplet_train
            image_dataset = self.data_manager.image_train
        else:
            triplet_dataset = self.data_manager.triplet_test
            image_dataset = self.data_manager.image_test
        # choose a random refererence from the triplet dataset
        random_index_index = randrange(triplet_dataset.__len__()) # index in an index dataset
        reference_triplet = triplet_dataset.__getitem__(random_index_index)
        self.reference_index = reference_triplet[0]
        # set reference data for the localizer
        self.reference_data = image_dataset.__getitem__(self.reference_index)
        self.oracle.set_reference(self.reference_index)
    
    """
        Runs the localization rollout experiment
    """
    def run(self, num_queries=20):
        # iterate through queries
        for query_index in range(num_queries):
            query = self.localizer.get_query()
            answer_query = sef.oracle.answer_query(query)

    

