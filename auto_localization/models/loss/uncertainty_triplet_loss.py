import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence

"""
    Triplet Loss that takes into account the uncertainty present in estimates 
    when calculating loss
"""
class UncertaintyTripletLoss(nn.Module):

    def __init__(self, kl_beta = 1.0, triplet_beta = 1.0, latent_dim=2):
        super(UncertaintyTripletLoss, self).__init__()
        self.kl_beta = kl_beta
        self.triplet_beta = triplet_beta
        self.latent_dim = latent_dim
        # empty linear to appease nn module requirement
        self.empty_linear = nn.Linear(1, 1)
        self.device = "cuda"

    def jensen_shannon(self, prob_a, prob_b):
        # unpack batch parameters
        mean_a, logvar_a = prob_a
        mean_b, logvar_b = prob_b
        # convert the batch parameters 
        cov_a = torch.diag_embed(logvar_a.exp(), dim1=-2, dim2=-1)
        a_dist = MultivariateNormal(mean_a, cov_a)
        cov_b = torch.diag_embed(logvar_b.exp(), dim1=-2, dim2=-1)
        b_dist = MultivariateNormal(mean_b, cov_b)
        # calculate average of two distributions
        average_dist_mean = 1/2*(mean_a + mean_b)
        average_dist_cov = 1/4*(cov_a + cov_b)
        average_dist = MultivariateNormal(average_dist_mean, average_dist_cov)
        # calculate the Jensen Shannon Divergence
        a_average = kl_divergence(a_dist, average_dist)
        b_average = kl_divergence(b_dist, average_dist)
        js = 1/2*(a_average + b_average)
        js = js[:, None]
        return -1*js.to("cuda")

    """
        This is a derivation of triplet loss incorporating the posterior
        probability distributions made by the encoder. This is from the 
        Bayesian representation learning with oracle constraints paper. 
    """
    def triplet_loss(self, anchor, positive, negative):
        # instead of doing triplet loss calculate the probability of triplets
        # being satisfied according to a bernoulli response model
        dist_anchor_positive = self.jensen_shannon(anchor, positive)
        dist_anchor_negative = self.jensen_shannon(anchor, negative)
        # softmax response
        dists = torch.cat((dist_anchor_positive, dist_anchor_negative), 1)
        probability = F.softmax(dists, 1)[:, 0]
        # negative is because we are minimizing this 
        loss = -1*probability 
        loss = torch.mean(loss)
        return loss

    def reconstruction_loss(self, real_data, fake_data):
        rec_loss = F.binary_cross_entropy(fake_data, real_data, reduction='mean')
        return rec_loss

    def kl_divergence_loss(self, mean, logvar):
        kl_div = -.5 * (1. + logvar - mean ** 2 - logvar.exp()).mean()
        return kl_div

    def forward(self, real_data, fake_data, mean, logvar, triplet_data, **kwargs):
        # latent vectors of the anchor, positive, and negative data
        anchor, positive, negative = triplet_data

        if not self.triplet_beta == 0.0:
            triplet_error = self.triplet_loss(anchor, positive, negative)
        else:
            triplet_error = torch.Tensor([0.0]).to(self.device)
        recon_error = self.reconstruction_loss(real_data, fake_data)
        kl_divergence_error = self.kl_divergence_loss(mean, logvar)

        final_loss = recon_error + self.kl_beta * kl_divergence_error + self.triplet_beta * triplet_error
        return final_loss, triplet_error, recon_error, kl_divergence_error

 
