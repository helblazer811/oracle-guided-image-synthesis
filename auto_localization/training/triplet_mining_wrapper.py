import torch
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data.sampler import Sampler
import sys
sys.path.append("../..")
from auto_localization.localization.noise_model_selector import NoiseModelSelector
import numpy as np
import random

class TripletMiningBatchSampler(Sampler):
    
    def __init__(self, model, triplet_dataset, batch_size, sample_rate=0.1):
        self.model = model
        self.triplet_dataset = triplet_dataset
        self.image_dataset = self.triplet_dataset.image_dataset
        self.metadata_dataset = self.triplet_dataset.metadata_dataset
        self.oracle = self.triplet_dataset.oracle
        self.batch_size = batch_size
        self.which_digits = self.triplet_dataset.which_digits
        self.sample_rate = sample_rate
        self.indexed = False
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.noise_model_selector = NoiseModelSelector(
            self.model, 
            num_triplets=self.samples_per, 
            triplet_dataset=self.triplet_dataset, 
            localizer_type="RandomLogistic"
        )

    def _fast_unique_triplets(self, index):
        triplets = []
        for i in range(int(batch_size / self.sample_rate)):
            triplet, _ = self.triplet_dataset[i]
            triplet = torch.stack([triplet[0], triplet[1], triplet[2]], dim=0)
            triplets.append(triplet)
        triplets = torch.stack(triplets).cuda()
        return triplets

    def _select_low_probability(self, triplets):
        # evaluate the triplets
        triplets = self.noise_model_selector.evaluate_triplets(triplets=triplets)
        # evaluate the logistic response model
        best_setting = {"k": 1.0, "normalization": 0}
        logistic_probs = self.noise_model_selector.compute_success_probabilities(best_setting, triplets)
        logistic_probs = np.stack(logistic_probs).squeeze()
        # select the query with the lowest probs
        maximum_index = np.argmax(logistic_probs)
        #anchor_index, positive_index, negative_index = 

        return (anchor_index, positive_index, negative_index)
       
    def __iter__(self, index):
        # print("wrapper get item")
        # get unique triplets
        triplets = self._fast_unique_triplets(index)
        # select the best one
        anchor, positive, negative = self._select_low_probability(triplets)

        return (anchor, positive, negative, -1), []

    def __len__(self):
        return len(self.triplet_dataset)

class TripletMiningDatasetWrapper(Dataset):
    
    def __init__(self, model, triplet_dataset, samples_per=10, margin=1.0):
        self.model = model
        self.triplet_dataset = triplet_dataset
        self.image_dataset = self.triplet_dataset.image_dataset
        self.metadata_dataset = self.triplet_dataset.metadata_dataset
        self.label_to_indices = self.triplet_dataset.label_to_indices
        self.oracle = self.triplet_dataset.oracle
        self.samples_per = samples_per
        self.margin = margin
        self.which_digits = self.triplet_dataset.which_digits
        self.indexed = False
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.noise_model_selector = NoiseModelSelector(
            self.model, 
            num_triplets=self.samples_per, 
            triplet_dataset=self.triplet_dataset, 
            localizer_type="RandomLogistic"
        )

    def _get_unique_triplets(self, index):
        #print("get unique triplets")
        anchor_label = self.image_dataset.labels[index].item()
        # get two different random images
        index_a = index
        index_b = index
        none_equal = False
        while not none_equal:
            index_a = np.random.choice(self.label_to_indices[anchor_label], size=self.samples_per, replace=False)
            # test if any indices are equal
            list_of_equal = np.nonzero(np.equal(index_a, index))[0]
            none_equal = np.shape(list_of_equal)[0] == 0
        none_equal = False
        while not none_equal:
            index_b = np.random.choice(self.label_to_indices[anchor_label], size=self.samples_per, replace=False)
            # test if any indices are equal
            list_of_equal = np.nonzero(np.equal(index_b, index))[0]
            none_equal = np.shape(list_of_equal)[0] == 0
            list_of_equal = np.nonzero(np.equal(index_b, index_a))[0]
            none_equal = none_equal and np.shape(list_of_equal)[0] == 0
        # make a triplet based on their relative morpho_distances
        #print("attempted batched oracle answering")
        choice = self.oracle.answer_query((index, index_a, index_b)) # oracle assumed to be IndexedMetadataOracle
        #print("oracle choices")
        #print(choice)
        positive = np.where(choice == 0, index_a, index_b) # index_a if choice == 0 else index_b
        negative = np.where(choice == 0, index_b, index_a) # index_b if choice == 0 else index_a
        # images from indices
        #print("attempt batched image access")
        anchor = self.image_dataset[index]
        # repeat anchor
        anchor = anchor.unsqueeze(1)
        anchor = torch.repeat_interleave(anchor, self.samples_per, dim=1)
        #print(anchor.shape)
        positive = self.image_dataset[positive]
        negative = self.image_dataset[negative]
        stacked_out = torch.stack((anchor, positive, negative), dim=0)    
        stacked_out = stacked_out.to(self.device)
        stacked_out = stacked_out.squeeze()
        stacked_out = stacked_out.unsqueeze(2)
        stacked_out = stacked_out.permute(1, 0, 2, 3, 4)
        #print(stacked_out.shape) 
        return stacked_out

    def _fast_unique_triplets(self):
        triplets = []
        indices = np.random.choice(np.arange(len(self.triplet_dataset)), size=self.samples_per)
        for i in indices:
            triplet, _ = self.triplet_dataset[i]
            triplet = torch.stack([triplet[0], triplet[1], triplet[2]], dim=0)
            triplets.append(triplet)
        triplets = torch.stack(triplets).cuda()
        return triplets

    def _select_hard_negative(self, triplets):
        # does hard negative mining on the given triplets        
        embedded_anchor, _, _, _ = self.model(triplets[0][0])
        # embed each of the negatives
        negatives = triplets[:, 2, :, :, :].squeeze()
        embedded_negatives, _, _, _ = self.model(negatives)
        # this score represents the squared distance between an anchor and a negative image in latent space        
        hard_negative_score = torch.norm(embedded_anchor - embedded_negatives, dim=1)
        # get the minimum of ths score
        minimum_index = torch.argmin(hard_negative_score)
        # make dataset references
        anchor, positive, negative = triplets[minimum_index][0], triplets[minimum_index][1], triplets[minimum_index][2]
        
        return anchor, positive, negative

    def _select_hard_positive(self, triplets):
        # does hard negative mining on the given triplets        
        embedded_anchor, _, _, _ = self.model(triplets[0][0])
        # embed each of the negatives
        positives = triplets[:, 1, :, :, :].squeeze()
        embedded_positives, _, _, _ = self.model(positives)
        # this score represents the squared distance between an anchor and a negative image in latent space        
        hard_positive_score = torch.norm(embedded_anchor - embedded_positives, dim=1)
        # get the minimum of ths score
        maximum_index = torch.argmax(hard_positive_score)
        # make dataset references
        anchor, positive, negative = triplets[maximum_index][0], triplets[maximum_index][1], triplets[maximum_index][2]
        
        return anchor, positive, negative

    def _select_low_probability(self, triplets):
        # evaluate the triplets
        triplets = self.noise_model_selector.evaluate_triplets(triplets)
        # evaluate the logistic response model
        best_setting = {"k": 1.0, "normalization": 0}
        logistic_probs = self.noise_model_selector.compute_success_probabilities(best_setting, triplets)
        logistic_probs = np.stack(logistic_probs).squeeze()
        # select the query with the lowest probs
        maximum_index = np.argmax(logistic_probs)
        # make dataset references
        triplet, _ = self.triplet_dataset[maximum_index]
        anchor, positive, negative = triplet[0], triplet[1], triplet[2]

        return anchor, positive, negative

    def __getitem__(self, index):
        # print("wrapper get item")
        # get unique triplets
        triplets = self._fast_unique_triplets()
        # select the best one
        anchor, positive, negative = self._select_low_probability(triplets)
        """
        if random.random() > 0.5:
            anchor, positive, negative = self._select_hard_negative(triplets)
        else:
            anchor, positive, negative = self._select_hard_positive(triplets)
        """

        return (anchor, positive, negative, -1), []
 
    def __len__(self):
        return len(self.triplet_dataset)
