import random
import torch
import numpy as np
import sys
sys.path.append("..")
from localization.automatic_rollout import AutomaticRollout

"""
    This class handle an AutomaticRollout with an Oracle
""" 
class MetricAutomaticRollout(AutomaticRollout):

    def __init__(self, queries=20, oracle=None, metadata_oracle=None, image_dataset=None, localizer=None, model=None, reference_image=None):
        self.oracle = oracle
        self.metadata_oracle = metadata_oracle
        self.queries = queries
        self.image_dataset = image_dataset
        self.localizer = localizer
        self.model = model
        self.reference_image = reference_image
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.reference_image is None:
            self._setup_reference_image()

    def _setup_reference_image(self):
        # randomly choose a reference data from the dataset
        index = random.randint(0, len(self.image_dataset))
        self.reference_image = self.image_dataset[index]
        # set embedded reference
        embedded_reference, _ = self.model.encode(self.reference_image.to(self.device))
        embedded_reference = embedded_reference.squeeze().detach().cpu().numpy()
        # convert reference_image to image format
        self.reference_image = self.reference_image.squeeze()
        # add them to the localizer 
        self.localizer.embedded_reference = embedded_reference
        self.localizer.reference_data = self.reference_image


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

        return diffs

    """
        Measures the differences between the metadata vectors of the 
        posterior mean and reference at each time. 
    """
    def measure_localizer_metadata_loss(self):
        reference_metadata = self.metadata_oracle.calculate_weighted_metadata(self.reference_image)
        posterior_means = self.localizer.posterior_means
        diffs = []

        for mean in posterior_means:
            tc_mean = torch.Tensor(mean).to("cuda")
            tc_mean = tc_mean.unsqueeze(0)
            decoded_mean = self.model.decode(tc_mean).detach().cpu().numpy()
            decoded_mean = decoded_mean.squeeze().squeeze()
            mean_metadata = self.metadata_oracle.calculate_weighted_metadata(decoded_mean)
            diff = np.linalg.norm(mean_metadata - reference_metadata)
            diffs.append(diff)

        return diffs

    """
        Calculates statistics to evaluate how well localization worked
    """
    def get_localization_metrics(self):
        metadata_loss = self.measure_localizer_metadata_loss()     
        localization_loss = self.measure_localizer_reference_loss()
        auc_loss = self.measure_localizer_reference_auc()
        
        return metadata_loss, localization_loss, auc_loss

    """
        Takes the localizer that has been initialized and runs it for
        a number of queries
    """
    def run_localization(self):
        # perform rollout
        print("performing rollout")
        for query_index in range(self.queries):
            item_a, item_b = self.localizer.get_query()
            item_a = torch.Tensor(item_a)
            item_b = torch.Tensor(item_b)
            query = (self.reference_image, item_a, item_b)
            query_answer = self.metadata_oracle.answer_query(query)
            self.localizer.save_choice(query_answer)

