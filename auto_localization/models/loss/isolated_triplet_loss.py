import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
import numpy as np

"""
    Isolated Subspace Triplet Loss
"""
class IsolatedTripletLoss(nn.Module):

    def __init__(self, kl_beta = 1.0, triplet_beta = 1.0, triplet_margin = 1.0, latent_dim=2, similarity_dim=2, reconstructive_dim=2, bce=False, squared_triplet=False, recon_beta=1.0, **kwargs):
        super(IsolatedTripletLoss, self).__init__()
        self.device = "cuda"
        self.kl_beta = kl_beta
        self.triplet_beta = triplet_beta
        self.recon_beta = recon_beta
        self.squared_triplet = squared_triplet
        self.triplet_margin = triplet_margin
        self.latent_dim = latent_dim
        self.similarity_dim = similarity_dim
        self.reconstructive_dim  = reconstructive_dim
        # dummy layer to appease nn module requirements
        self.empty_linear = nn.Linear(1, 1)
        self.triplet_margin_loss = nn.TripletMarginLoss(margin=self.triplet_margin)
        self.bce = bce
    
    """
        Measures triplet loss
    """
    def triplet_loss(self, anchor, positive, negative, **kwargs):
        if not self.squared_triplet:
            return F.triplet_margin_loss(anchor, positive, negative, margin=self.triplet_margin)
        else:
            # calculate mahalanobis distances
            distance_anchor_positive = torch.norm(anchor - positive, dim=-1)**2
            distance_anchor_negative = torch.norm(anchor - negative, dim=-1)**2
            # calculate the triplet loss
            no_relu = True
            if no_relu:
                triplet_loss = distance_anchor_positive - distance_anchor_negative + self.triplet_margin
            else:
                triplet_loss = F.relu(distance_anchor_positive - distance_anchor_negative + self.triplet_margin)
            mean_loss = torch.mean(triplet_loss)
            return mean_loss
   
    def reconstruction_loss(self, real_data, fake_data):
        if self.bce:
            rec_loss = F.binary_cross_entropy(fake_data, real_data, reduction='mean')
            return rec_loss
        else:
            rec_loss = F.mse_loss(fake_data, real_data, reduction="mean")
            return rec_loss

    def kl_divergence_loss(self, mean, logvar):
        kl_div = -.5 * (1. + logvar - mean ** 2 - logvar.exp()).mean()
        return kl_div

    def triplet_percentage(self, anchor, positive, negative):
        # calculate distances
        distance_anchor_positive = torch.norm(anchor - positive, dim=-1)
        distance_anchor_negative = torch.norm(anchor - negative, dim=-1)
        # test if it is negative
        num_closer = torch.sum((distance_anchor_positive < distance_anchor_negative).int())
        percentage = torch.Tensor([num_closer/anchor.shape[0]])
        return percentage

    def forward(self, real_data, fake_data, mean, logvar, triplet_data, **kwargs):
        test_mode = False
        if "test_mode" in kwargs:
            test_mode = kwargs["test_mode"]
        # latent vectors of the anchor, positive, and negative data
        if len(triplet_data) == 4:
            anchor, positive, negative, attribute_index = triplet_data
        else:
            anchor, positive, negative = triplet_data
        # unpack triplet data from logvars
        anchor, _ = anchor
        positive, _ = positive
        negative, _ = negative
        # segment the triplet
        anchor = anchor[:, :self.similarity_dim]
        positive = positive[:, :self.similarity_dim]
        negative = negative[:, :self.similarity_dim]
        if test_mode or not self.triplet_beta == 0.0:
            triplet_error = self.triplet_loss(anchor, positive, negative)
        else:
            triplet_error = torch.Tensor([0.0]).to(self.device)
        recon_error = self.reconstruction_loss(real_data, fake_data)
        kl_divergence_error = self.kl_divergence_loss(mean, logvar)
        final_loss = self.recon_beta * recon_error + self.kl_beta * kl_divergence_error + self.triplet_beta * triplet_error
        triplet_percentage = self.triplet_percentage(anchor, positive, negative)

        loss_dict = {
            "loss": final_loss,
            "triplet_loss": triplet_error,
            "recon_loss": recon_error,
            "kl_divergence_loss": kl_divergence_error,
            "triplet_percentage": triplet_percentage
        }

        return loss_dict
