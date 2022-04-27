import torch
from torch import nn
import torch.nn.functional as F
import geotorch
import numpy as np

"""
    VAE Triplet Loss module 
"""
class VAEMetricTripletLoss(nn.Module):

    def __init__(self, kl_beta = 1.0, triplet_beta = 1.0, triplet_margin = 1.0, latent_dim=2):
        super(VAEMetricTripletLoss, self).__init__()
        self.kl_beta = kl_beta
        self.triplet_beta = triplet_beta
        self.triplet_margin= triplet_margin
        self.latent_dim = latent_dim
        # linear mahalanobis M matrix
        self.metric_linear = nn.Linear(self.latent_dim, self.latent_dim)
        self.inverse_metric_linear = nn.Linear(self.latent_dim, self.latent_dim)
        self.inverse_metric_linear.weight = nn.Parameter(torch.eye(self.latent_dim))
        # constrain metric_linear to be positive semi-definite
        geotorch.positive_semidefinite(self.metric_linear)
    
    def setup_inverse(self):
        weights = self.metric_linear.weight
        inverse_mat = torch.inverse(weights)
        self.inverse_metric_linear.weight = nn.Parameter(inverse_mat)

    def mahalanobis_distance(self, a, b):
        diff = a - b
        batch_size = a.shape[0]
        distance = torch.bmm(diff.view(batch_size, 1, -1), self.metric_linear(diff).view(batch_size, -1, 1))
        #print(np.all(np.linalg.eigvals(self.metric_linear.weight.detach().cpu().numpy()) > 0))
        sqrt_dist = torch.sqrt(distance)
        sqrt_dist = sqrt_dist.squeeze()
        return distance 

    def triplet_loss(self, triplet_data):
        anchor, positive, negative = triplet_data
        # unsqueeze dims
        anchor = anchor.unsqueeze(0)
        positive = positive.unsqueeze(0)
        negative = negative.unsqueeze(0)
        # calculate mahalanobis distances
        distance_anchor_positive = self.mahalanobis_distance(anchor, positive)
        distance_anchor_negative = self.mahalanobis_distance(anchor, negative)
        # calculate the triplet loss
        triplet_loss = F.relu(distance_anchor_positive - distance_anchor_negative + self.triplet_margin)
        mean_loss = torch.mean(triplet_loss)
        return mean_loss
    
    """
        Measures triplet loss without the mahalanobis distance
    """
    def identity_triplet_loss(self, triplet_data):
        anchor, positive, negative = triplet_data
        # unsqueeze dims
        anchor = anchor.unsqueeze(0)
        positive = positive.unsqueeze(0)
        negative = negative.unsqueeze(0)
        # calculate mahalanobis distances
        distance_anchor_positive = torch.norm(anchor - positive, dim=-1)
        distance_anchor_negative = torch.norm(anchor - negative, dim=-1)
        # calculate the triplet loss
        triplet_loss = F.relu(distance_anchor_positive - distance_anchor_negative + self.triplet_margin)
        mean_loss = torch.mean(triplet_loss)
        return mean_loss
    
    def reconstruction_loss(self, real_data, fake_data):
        rec_loss = F.binary_cross_entropy(fake_data, real_data, reduction='mean')
        return rec_loss

    def kl_divergence_loss(self, mean, logvar):
        kl_div = -.5 * (1. + logvar - mean ** 2 - logvar.exp()).mean()
        return kl_div

    def forward(self, real_data, fake_data, mean, logvar, triplet_data, **kwargs):
        # latent vectors of the anchor, positive, and negative data
        anchor, positive, negative = triplet_data
        anchor, positive, negative = anchor[0], positive[0], negative[0] # unpack means and ignore logvars
        
        if not self.triplet_beta == 0.0:
            triplet_error = self.triplet_loss((anchor, positive, negative))
        else:
            triplet_error = torch.Tensor([0.0]).to(self.device)
        recon_error = self.reconstruction_loss(real_data, fake_data)
        kl_divergence_error = self.kl_divergence_loss(mean, logvar)

        final_loss = recon_error + self.kl_beta * kl_divergence_error + self.triplet_beta * triplet_error
        return final_loss, triplet_error, recon_error, kl_divergence_error

