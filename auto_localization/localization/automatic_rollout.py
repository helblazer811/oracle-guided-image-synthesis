import wandb
import random
import torch
import numpy as np
from scipy.spatial import distance
import sys

"""
    This class handle an AutomaticRollout with an Oracle
""" 
class AutomaticRollout():

    def __init__(self, queries=20, oracle=None, metadata_oracle=None, image_dataset=None, localizer=None, model=None, reference_image=None, reference_metadata=None, metadata_dataset=None, batched=False):
        self.oracle = oracle
        self.metadata_oracle = metadata_oracle
        self.queries = queries
        self.image_dataset = image_dataset
        self.batched = batched
        self.localizer = localizer
        self.model = model
        self.reference_image = reference_image
        self.reference_metadata = reference_metadata
        self.metadata_dataset = metadata_dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # setup reference image
        self._setup_reference_image(image=reference_image, metadata=reference_metadata)

    def _setup_reference_image(self, image=None, metadata=None):
        if image is None:
            # randomly choose a reference data from the dataset
            index = random.randint(0, len(self.image_dataset) - 1)
            self.reference_image = self.image_dataset[index]
            self.reference_metadata = self.metadata_oracle.metadata_dataset[index]
        else:
            self.reference_image = image
            self.reference_metadata = metadata
        # set embedded reference
        embedded_reference, _ = self.model.encode(self.reference_image.to(self.device))
        embedded_reference = embedded_reference.squeeze().detach().cpu().numpy()
        # convert reference_image to image format
        self.reference_image = self.reference_image.squeeze()
        # add them to the localizer 
        self.localizer.embedded_reference = embedded_reference
        self.localizer.reference_data = self.reference_image

    """
        Logs rollout
    """
    def log_rollout(self, number=0):
        # get metrics
        metrics = self.get_localization_metrics()
        metadata_metrics = metrics["metadata_loss"]
        latent_metrics = metrics["latent_loss"]
        # convert to numpy 
        metadata_metrics = np.array(metadata_metrics)
        latent_metrics = np.array(latent_metrics)
        time_vals = np.arange(0, np.shape(metadata_metrics)[0])
        # log the metrics
        metadata = np.concatenate((time_vals[:, None], metadata_metrics[:, None]), axis=1)
        wandb.log({f"metadata_metrics_{number}": wandb.Table(data=metadata,
                                columns = ["time_vals", "metadata_metrics"])})
        time_vals = np.arange(0, np.shape(latent_metrics)[0])
        latent = np.concatenate((time_vals[:, None], latent_metrics[:, None]), axis=1)
        wandb.log({f"latent_metrics_{number}": wandb.Table(data=latent,
                                columns = ["time_vals", "latent_metrics"])})
  
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
        Function to calculate the reference percentiles using the 
        mahalanobis distance, which incorporates the posterior covariance
    """
    def measure_reference_percentiles(self):
        # calculate the mahalnobis distance of each point compared to the posterior
        embedding = self.localizer.embedding
        #for embedding_point in localizer_0.embedding
        def calculate_mahalanobis_percentile(mean, covariance, embedded_reference):
            # check if matrix is singular
            if np.linalg.cond(covariance) < 1/sys.float_info.epsilon:
                inv_cov = np.linalg.inv(covariance)
            else:
                return 0
            distances = []
            for point in embedding:
                # calculate the mahalanobis distance of each point from the posterior mean
                mahalanobis_distance = distance.mahalanobis(mean, point, inv_cov)
                distances.append(mahalanobis_distance)
            distances = np.array(distances)
            distances = np.sort(distances)
            # calculate the percentile of the embedded_reference
            ref_distance = distance.mahalanobis(mean, embedded_reference, inv_cov)
            percentile = np.searchsorted(distances, ref_distance) / np.shape(distances)[0]
            
            return percentile

        def calculate_mahalanobis_percentiles():
            percentiles = []
            for iteration in range(len(self.localizer.posterior_means)):
                percentile = calculate_mahalanobis_percentile(self.localizer.posterior_means[iteration], self.localizer.vars[iteration], self.localizer.embedded_reference)
                percentiles.append(percentile)
            return percentiles
                
        #percentiles = calculate_mahalanobis_percentiles()
        percentiles *= 100
        return percentiles

    """
        Measures the percentile of how close a posterior mean is to the 
        reference in metadata space 
    """
    def measure_reference_percentiles_deprecated(self):
        percentiles = []
        # Get the test images and metadata vectors
        test_batch_size = 50
        random_indices = np.random.randint(0, len(self.image_dataset), size=test_batch_size)# generate indices
        test_images = self.image_dataset[random_indices]  
        test_metadata = self.metadata_oracle.metadata_dataset[random_indices]
        if hasattr(self.metadata_oracle, "component_weighting"):
            test_metadata = self.metadata_oracle.component_weighting * test_metadata
        if isinstance(test_metadata, torch.Tensor):
            test_metadata = test_metadata.detach().cpu().numpy()
        if self.reference_metadata is None:
            reference_metadata = self.metadata_oracle.calculate_weighted_metadata(self.reference_image)
        else:
            reference_metadata = self.reference_metadata
        if isinstance(reference_metadata, torch.Tensor):
            reference_metadata = reference_metadata.detach().cpu().numpy()

        # Go through each mean 
        for posterior_mean in self.localizer.posterior_means:
            posterior_mean = torch.Tensor(posterior_mean).to("cuda")
            posterior_mean = posterior_mean.unsqueeze(0)
            decoded_mean = self.model.decode(posterior_mean).detach().cpu().numpy()
            posterior_mean_image = decoded_mean.squeeze().squeeze()
            # Get the most recent posterior mean and corresponding metadata
            posterior_mean_metadata = self.metadata_oracle.calculate_weighted_metadata(posterior_mean_image)
            if isinstance(posterior_mean_metadata, torch.Tensor):
                posterior_mean_metadata = posterior_mean_metadata.detach().cpu().numpy()
            posterior_mean_distance = np.linalg.norm(reference_metadata - posterior_mean_metadata)
            # Calculate the distance between each metadata vector and the reference metadata vector
            metadata_distances = np.linalg.norm(reference_metadata - test_metadata, axis=1)
            # Sort the metadata distances 
            sorted_distances = np.sort(metadata_distances)
            # Find the place of the posterior_mean_meatadata in that distance
            insert_index = np.searchsorted(sorted_distances, [posterior_mean_distance])[0]
            # Calculate what percentile the ideal point estimate's metadata vector is at in terms of 
            percentile = 100.0 * ((insert_index)/test_batch_size)
            percentiles.append(percentile)
        return percentiles

    """
        Measures the differences between the metadata vectors of the 
        posterior mean and reference at each time. 
    """
    def measure_localizer_metadata_loss(self):
        reference_metadata = self.metadata_oracle.calculate_weighted_metadata(self.reference_image.cuda())
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
        Measures the determinant of the covariance matrix at a given point in time
    """
    def measure_localizer_latent_covariance_determinant(self):
        variances = self.localizer.vars.copy()
        determinants = []
        for var in variances:
            determinants.append(np.linalg.det(var))
            
        return determinants

    """
        Measures the differences between the metadata vectors of the 
        posterior mean and reference at each time. 
    """
    def measure_broken_down_metadata_loss(self):
        reference_metadata = self.metadata_oracle.calculate_weighted_metadata(self.reference_image)
        if isinstance(reference_metadata, torch.Tensor):
            reference_metadata = reference_metadata.detach().cpu().numpy()
        posterior_means = self.localizer.posterior_means.copy()
        posterior_means.insert(0, np.mean(self.localizer.embedding, axis=0))
        metadata_losses = []

        for mean in posterior_means:
            tc_mean = torch.Tensor(mean).to("cuda")
            tc_mean = tc_mean.unsqueeze(0)
            decoded_mean = self.model.decode(tc_mean).detach().cpu().numpy()
            decoded_mean = decoded_mean.squeeze().squeeze()
            mean_metadata = self.metadata_oracle.calculate_weighted_metadata(decoded_mean)
            if isinstance(mean_metadata, torch.Tensor):
                mean_metadata = mean_metadata.detach().cpu().numpy()
           
            losses = np.abs(mean_metadata - reference_metadata)

            metadata_losses.append(losses)
        
        # numpy array with columns [area, length, thickness, slant, width, height]
        return metadata_losses

    def measure_nearest_neighbor_metadata_loss(self, sample_size=500):
        reference_metadata = self.metadata_oracle.calculate_weighted_metadata(self.reference_image.cuda())
        if isinstance(reference_metadata, torch.Tensor):
            reference_metadata = reference_metadata.detach().cpu().numpy()
        posterior_means = self.localizer.posterior_means.copy()
        diffs = []
        self.model.similarity_mode = True
        self.model.eval()
        # sample a bunch of images and calculate their metadata vectors
        image_indices = np.random.choice(len(self.image_dataset), size=(sample_size))
        similarity_vectors = []
        for index in image_indices:
            image = self.image_dataset[index].cuda()
            similarity_vector, _, _, _ = self.model.forward(image)
            similarity_vectors.append(similarity_vector)

        similarity_vectors = torch.stack(similarity_vectors).cuda()

        def get_closest_image(similarity_vector):
            distances = torch.norm(similarity_vector - similarity_vectors, dim=-1)
            closest_index = torch.argmin(distances)
            closest_image = self.image_dataset[image_indices[closest_index]]
            return closest_image

        for mean in posterior_means:
            tc_mean = torch.Tensor(mean).to("cuda")
            tc_mean = tc_mean.unsqueeze(0)
            nearest_image = get_closest_image(tc_mean)
            mean_metadata = self.metadata_oracle.calculate_weighted_metadata(nearest_image)
            if isinstance(mean_metadata, torch.Tensor):
                mean_metadata = mean_metadata.detach().cpu().numpy()
            diff = np.linalg.norm(mean_metadata - reference_metadata)
            diffs.append(diff)

        return diffs

    def measure_nearest_neighbor_percentile(self, sample_size=1000):
        percentiles = []
        # Get the test images and metadata vectors
        random_indices = np.random.randint(0, len(self.image_dataset), size=sample_size)# generate indices
        test_metadata = []
        for index in random_indices:
            if isinstance(self.metadata_oracle.metadata_dataset[index], np.ndarray):
                test_metadata.append(self.metadata_oracle.metadata_dataset[index])
            else:
                test_metadata.append(self.metadata_oracle.metadata_dataset[index].detach().cpu().numpy())
        test_metadata = np.stack(test_metadata).squeeze()
        if hasattr(self.metadata_oracle, "component_weighting"):
            test_metadata = self.metadata_oracle.component_weighting * test_metadata
        if isinstance(test_metadata, torch.Tensor):
            test_metadata = test_metadata.detach().cpu().numpy()
        #reference_metadata = self.metadata_oracle.calculate_weighted_metadata(self.reference_image)
        if hasattr(self.metadata_oracle, "component_weighting"):
            reference_metadata = self.reference_metadata * self.metadata_oracle.component_weighting
        else:
            reference_metadata = self.reference_metadata
        if isinstance(reference_metadata, torch.Tensor):
            reference_metadata = reference_metadata.detach().cpu().numpy()

        # sample a bunch of images and calculate their metadata vectors
        similarity_vectors = []
        for index in random_indices:
            image = self.image_dataset[index].cuda()
            similarity_vector, _, _, _ = self.model.forward(image)
            similarity_vectors.append(similarity_vector)

        similarity_vectors = torch.stack(similarity_vectors).cuda()

        def get_closest_image(similarity_vector):
            distances = torch.norm(similarity_vector - similarity_vectors, dim=-1)
            closest_index = torch.argmin(distances)
            closest_image = self.image_dataset[random_indices[closest_index]]
            if hasattr(self.metadata_oracle, "component_weighting"):
                closest_metadata = self.metadata_dataset[random_indices[closest_index]] * self.metadata_oracle.component_weighting
            else:
                closest_metadata = self.metadata_dataset[random_indices[closest_index]]
            return closest_image, closest_metadata

        # Go through each mean 
        for posterior_mean in self.localizer.posterior_means:
            posterior_mean = torch.Tensor(posterior_mean).to("cuda")
            posterior_mean = posterior_mean.unsqueeze(0)
            decoded_mean = self.model.decode(posterior_mean).detach().cpu().numpy()
            posterior_mean_image = decoded_mean.squeeze().squeeze()
            # Get the most recent posterior mean and corresponding metadata
            nearest_image, nearest_metadata = get_closest_image(posterior_mean)
            if isinstance(nearest_metadata, torch.Tensor):
                nearest_metadata = nearest_metadata.detach().cpu().numpy()
            nearest_distance = np.linalg.norm(nearest_metadata - reference_metadata)
            # Calculate the distance between each metadata vector and the reference metadata vector
            metadata_distances = np.linalg.norm(reference_metadata - test_metadata, axis=1)
            # Sort the metadata distances 
            sorted_distances = np.sort(metadata_distances)
            # Find the place of the posterior_mean_meatadata in that distance
            insert_index = np.searchsorted(sorted_distances, [nearest_distance])[0]
            # Calculate what percentile the ideal point estimate's metadata vector is at in terms of 
            percentile = 100.0 * ((insert_index)/sample_size)
            percentiles.append(percentile)

        return percentiles

    """
        Calculates statistics to evaluate how well localization worked
    """
    def get_localization_metrics(self):
        metrics = {
            "metadata_loss": self.measure_localizer_metadata_loss(),
            "latent_covariance_determinant": self.measure_localizer_latent_covariance_determinant(),
            "broken_down_metadata_loss": self.measure_broken_down_metadata_loss(),
            "localization_loss": self.measure_localizer_reference_loss(),
            "auc_loss": self.measure_localizer_reference_auc(),
            "nearest_neighbor_loss": self.measure_nearest_neighbor_metadata_loss(),
            #"reference_percentiles": self.measure_reference_percentiles(),
            "nearest_neighbor_percentile": self.measure_nearest_neighbor_percentile()
        }
           
        return metrics

    """
        Batched version of run_localization
    """
    def run_batched_localization(self):
        import time
        # set model to be not training
        self.model.training = False
        # Sample posterior before queries
        self.localizer.get_posterior_samples()
        # perform rollout
        for query_index in range(self.queries):
            item_a, item_b = self.localizer.get_query()
            item_a = torch.Tensor(item_a)
            item_b = torch.Tensor(item_b)
            query = (self.reference_image, item_a, item_b)
            query_answer = self.oracle.answer_query(query)
            self.localizer.save_choice(query_answer)
        end = time.time()
        # Sample posterior after a query
        self.localizer.get_posterior_samples()

    """
        Takes the localizer that has been initialized and runs it for
        a number of queries
    """
    def run_localization(self, batched=False):
        if batched:
            self.run_batched_localization()
            return
        # set model to be not training
        self.model.training = False
        # perform rollout
        print("performing rollout")
        for query_index in range(self.queries):
            # Sample posterior after a query
            self.localizer.get_posterior_samples()
            # Get a query
            item_a, item_b = self.localizer.get_query()
            item_a = torch.Tensor(item_a)
            item_b = torch.Tensor(item_b)
            query = (self.reference_image, item_a, item_b)
            query_answer = self.oracle.answer_query(query)
            self.localizer.save_choice(query_answer)

