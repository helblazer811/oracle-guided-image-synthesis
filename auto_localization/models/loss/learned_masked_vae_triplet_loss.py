import torch
from torch import nn
import torch.nn.functional as F
import wandb

"""
    Masked VAE Triplet Loss module 
"""
class LearnedMaskedVAETripletLoss(nn.Module):
 
    def __init__(self, kl_beta = 1.0, triplet_beta = 1.0, triplet_margin = 1.0, mask_regularization_beta=1000.0, latent_dim = 2, similarity_dim=1, reconstructive_dim=1, attributes=[0, 1]):
        super(LearnedMaskedVAETripletLoss, self).__init__()
        self.kl_beta = kl_beta
        self.triplet_beta = triplet_beta
        self.triplet_margin = triplet_margin
        self.mask_regularization_beta = mask_regularization_beta
        self.latent_dim = latent_dim
        self.train_mode = True
        self.attributes = attributes
        self.similarity_dim = similarity_dim
        self.reconstructive_dim = reconstructive_dim
        # define the triplet mask vectors 
        self.eps = 1e-7
        self.sigmoid = nn.Sigmoid()
        self.device = "cuda"
        self._setup_masks()

    def _setup_masks(self):
        # masks is a linear layer of size (num_attributes, latent_dim)
        num_attributes = len(self.attributes)
        self.masks = nn.parameter.Parameter(torch.ones(num_attributes, self.latent_dim))
        # self.attribute_to_index maps the attribute number to the mask index in masks
        self.attribute_to_index = {}
        for index, attribute in enumerate(self.attributes):
            self.attribute_to_index[attribute] = index

    def triplet_loss(self, anchor, positive, negative, attribute_index=-1):
        # get the mask corresponding to the index
        mask_vectors = []
        for attribute_ind in attribute_index:
            mask_vectors.append(self.masks[self.attribute_to_index[int(attribute_ind)]])
        mask_vectors = torch.stack(mask_vectors)
        mask_vectors = mask_vectors.to(self.device)
        mask_vectors = mask_vectors.unsqueeze(1)
        # apply the mask to each of the latent dimensions
        positive_anchor_distance = torch.abs(anchor - positive).unsqueeze(2)
        positive_anchor = torch.bmm(mask_vectors, positive_anchor_distance).squeeze()
        negative_anchor_distance = torch.abs(anchor - negative).unsqueeze(2)
        negative_anchor = torch.bmm(mask_vectors, negative_anchor_distance).squeeze()
        # get the differences of distances
        masked_distance_difference = positive_anchor - negative_anchor
        # calculate the final masked triplet loss
        masked_triplet_loss = F.relu(masked_distance_difference + self.triplet_margin)
        # get mean
        masked_triplet_loss = torch.mean(masked_triplet_loss)
        return masked_triplet_loss

    def reconstruction_loss(self, real_data, fake_data):
        rec_loss = F.binary_cross_entropy(fake_data, real_data, reduction='mean')
        return rec_loss

    def kl_divergence_loss(self, mean, logvar):
        kl_div = -.5 * (1. + logvar - mean ** 2 - logvar.exp()).mean()
        return kl_div

    def mask_regularization_loss(self):
        return torch.mean(torch.norm(self.masks.data, p=1, dim=1))

    def forward(self, real_data, fake_data, mean, logvar, triplet_data, **kwargs):
        anchor, positive, negative, attribute_index = triplet_data
        if not self.triplet_beta == 0.0:
            anchor, positive, negative = anchor[0], positive[0], negative[0] # unpack means and ignore logvars
            triplet_error = self.triplet_loss(anchor, positive, negative, attribute_index)
        else:
            triplet_error = torch.Tensor([0.0]).to(self.device)
        recon_error = self.reconstruction_loss(real_data, fake_data)
        kl_divergence_error = self.kl_divergence_loss(mean, logvar)
        mask_regularization_loss = self.mask_regularization_loss()

        final_loss = recon_error + self.kl_beta * kl_divergence_error + self.triplet_beta * triplet_error + self.mask_regularization_beta * mask_regularization_loss
        return final_loss, triplet_error, recon_error, kl_divergence_error, mask_regularization_loss

 
