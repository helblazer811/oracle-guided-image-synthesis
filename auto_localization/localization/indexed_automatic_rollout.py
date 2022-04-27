import random
import torch
import numpy as np

"""
    This class handle an IndexedAutomaticRollout with an Oracle
    This differs from Automatic Rollout in that it only has access to
    indexed oracles and the indices of items. 
""" 
class IndexedAutomaticRollout():

    def __init__(self, queries=20, oracle=None, metadata_oracle=None, image_dataset=None, localizer=None, model=None, reference_index=None, reference_metadata=None):
        self.oracle = oracle
        self.metadata_oracle = metadata_oracle
        self.queries = queries
        self.image_dataset = image_dataset
        self.localizer = localizer
        self.model = model
        self.reference_index = reference_index
        self.reference_metadata = reference_metadata
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_reference_index(self.reference_index)

    def _setup_reference_index(self, reference_index=None):
        if reference_index is None:
            # randomly choose a reference data from the dataset
            index = random.randint(0, len(self.image_dataset) - 1)
            self.reference_index = index
            self.reference_metadata = self.metadata_oracle.metadata_dataset[index]
            self.reference_image = self.image_dataset[index]
        else:
            self.reference_index = reference_index
            self.reference_metadata = self.metadata_oracle.metadata_dataset[reference_index]
            self.reference_image = self.image_dataset[self.reference_index]

        self.reference_metadata = np.dot(self.reference_metadata, self.metadata_oracle.component_weighting.T)
        # set embedded reference
        embedded_reference, _= self.model.encode(self.reference_image.cuda())
        embedded_reference = embedded_reference.squeeze().detach().cpu().numpy()
        # add them to the localizer 
        self.localizer.embedded_reference = embedded_reference
        self.localizer.reference_data = self.reference_index

    def measure_localizer_reference_auc(self):
        reference_loss = self.measure_localizer_metadata_loss()
        # calculate reimman sum of loss
        return np.sum(reference_loss)

    """
        Measures the difference between the 
        posterior mean at each point in time and the embedded reference 
    """
    def measure_localizer_reference_loss(self):
        embedded_reference = self.localizer.embedded_reference
        posterior_means = self.localizer.posterior_means
        diffs = []

        for mean in posterior_means:
            diff = np.linalg.norm(mean - embedded_reference, axis=0)
            diffs.append(diff)
        diffs = np.array(diffs)
        return diffs

    def measure_nearest_neighbor_metadata_loss(self, sample_size=500):
        reference_metadata = self.reference_metadata
        if isinstance(reference_metadata, torch.Tensor):
            reference_metadata = reference_metadata.detach().cpu().numpy()
        posterior_means = self.localizer.posterior_means.copy()
        posterior_means.insert(0, np.mean(self.localizer.embedding, axis=0))
        diffs = []
        self.model.similarity_mode = True
        self.model.eval()
        # sample a bunch of images and calculate their metadata vectors
        image_indices = np.random.choice(len(self.image_dataset), size=(sample_size))
        similarity_vectors = []
        for index in image_indices:
            image = self.image_dataset[index].cuda()
            _, _, similarity_vector, _ = self.model.forward(image)
            similarity_vectors.append(similarity_vector)

        similarity_vectors = torch.stack(similarity_vectors).cuda()

        def get_closest_index(similarity_vector):
            distances = torch.norm(similarity_vector - similarity_vectors, dim=-1)
            closest_index = torch.argmin(distances)
            return closest_index

        for mean in posterior_means:
            tc_mean = torch.Tensor(mean).to("cuda")
            tc_mean = tc_mean.unsqueeze(0)
            nearest_index = get_closest_index(tc_mean)
        
            mean_metadata = self.metadata_oracle.calculate_weighted_metadata(nearest_index)
            if isinstance(mean_metadata, torch.Tensor):
                mean_metadata = mean_metadata.detach().cpu().numpy()
            diff = np.linalg.norm(mean_metadata - reference_metadata)
            diffs.append(diff)

        return diffs

    """
        Measures the determinant of the covariance matrix at a given point in time
    """
    def measure_localizer_latent_covariance_determinant(self):
        variances = self.localizer.vars.copy()
        determinants = []
        for var in variances:
            determinants.append(np.linalg.det(var))
            
        return determinants

    """
        Measures the difference between the 
        posterior mean at each point in time and the embedded reference 
    """
    def measure_localizer_reference_loss(self):
        embedded_reference = self.localizer.embedded_reference
        if isinstance(embedded_reference, torch.Tensor):
            embedded_reference = embedded_reference.detach().cpu().numpy()
        posterior_means = self.localizer.posterior_means.copy() 
        posterior_means.insert(0, np.mean(self.localizer.embedding, axis=0))
        diffs = []

        for mean in posterior_means:
            # if the model is an isolated subspace model calculate the difference in 
            # just the similarity space
            # also if the model comes with a global mask then weight the distance by the mask
            if isinstance(mean, torch.Tensor):
                mean = mean.detach().cpu().numpy()

            diff = np.linalg.norm(mean - embedded_reference, axis=0)
            diffs.append(diff)

        return diffs

    """
        Measures the differences between the metadata vectors of the 
        posterior mean and reference at each time. 
    """
    def measure_localizer_metadata_loss(self):
        reference_metadata = self.reference_metadata
        if isinstance(reference_metadata, torch.Tensor):
            reference_metadata = reference_metadata.detach().cpu().numpy()
        posterior_means = self.localizer.posterior_means.copy()
        posterior_means.insert(0, np.mean(self.localizer.embedding, axis=0))
        diffs = []

        for mean in posterior_means:
            tc_mean = torch.Tensor(mean).to("cuda")
            tc_mean = tc_mean.unsqueeze(0)
            decoded_mean = self.model.decode(tc_mean).detach().cpu().numpy()
            decoded_mean = decoded_mean.squeeze().squeeze()
            mean_metadata = self.metadata_oracle.calculate_weighted_metadata(decoded_mean)
            if isinstance(mean_metadata, torch.Tensor):
                mean_metadata = mean_metadata.detach().cpu().numpy()

            diff = np.linalg.norm(mean_metadata - reference_metadata)
            diffs.append(diff)

        return diffs
 
    """
        Calculates statistics to evaluate how well localization worked
    """
    def get_localization_metrics(self):
        
        metrics = {
            "metadata_loss": self.measure_nearest_neighbor_metadata_loss(),
            "latent_covariance_determinant": self.measure_localizer_latent_covariance_determinant(),
            "localization_loss": self.measure_localizer_reference_loss(),
            "nearest_neighbor_loss": self.measure_nearest_neighbor_metadata_loss(),
        }
       
        return metrics

    """
        Takes the localizer that has been initialized and runs it for
        a number of queries
    """
    def run_localization(self):
        # perform rollout
        print("performing rollout")
        for query_index in range(self.queries):
            item_a, item_b = self.localizer.get_query()
            query = (self.reference_index, item_a, item_b)
            query_answer = self.metadata_oracle.answer_query(query)
            query_answer = 1 if query_answer == 1 else 0
            self.localizer.save_choice(query_answer)

