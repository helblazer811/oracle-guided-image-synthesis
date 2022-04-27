import torch
from torch import nn
import torch.nn.functional as F

"""
    VAE Triplet Loss module 
"""
class VAETripletLoss(nn.Module):

    def __init__(self, recon_beta = 1.0, kl_beta = 1.0, triplet_beta = 1.0, triplet_margin = 1., recon_loss_type="mse"):
        super(VAETripletLoss, self).__init__()
        self.kl_beta = kl_beta
        self.triplet_beta = triplet_beta
        self.triplet_margin = triplet_margin
        self.recon_beta = recon_beta
        self.recon_loss_type = recon_loss_type
        # empty linear to appease nn module requirement
        self.empty_linear = nn.Linear(1, 1)
        self.device = "cuda"

    def triplet_loss(self, anchor, positive, negative, **kwargs):
        return F.triplet_margin_loss(anchor, positive, negative, margin=self.triplet_margin)

    def reconstruction_loss(self, real_data, fake_data):
        if self.recon_loss_type is "mse":
            rec_loss = F.mse_loss(fake_data, real_data, reduction="mean") 
        else:
            rec_loss = F.binary_cross_entropy(fake_data, real_data, reduction='mean')
        return rec_loss

    def kl_divergence_loss(self, mean, logvar):
        kl_div = -.5 * (1. + logvar - mean ** 2 - logvar.exp()).mean()
        return kl_div

    def triplet_percentage(self, anchor, positive, negative):
        positive_anchor_distance = torch.norm(anchor - positive)
        negative_anchor_distance = torch.norm(anchor - negative)
        # get the differences of distances
        masked_distance_difference = positive_anchor_distance - negative_anchor_distance
        # if it is negative then the triplet is passed
        is_negative = masked_distance_difference < 0
        if len(is_negative.shape) > 0:
            percentage = torch.sum(is_negative) / is_negative.shape[0]
        else:
            percentage = is_negative
        percentage = torch.Tensor([percentage])

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
        anchor, positive, negative = anchor[0], positive[0], negative[0] # unpack means and ignore logvars
        if test_mode or not self.triplet_beta == 0.0:
            triplet_error = self.triplet_loss(anchor, positive, negative)
        else:
            triplet_error = torch.Tensor([0.0]).to(self.device)
        recon_error = self.reconstruction_loss(real_data, fake_data)
        kl_divergence_error = self.kl_divergence_loss(mean, logvar)
        triplet_percentage = self.triplet_percentage(anchor, positive, negative)

        final_loss = recon_error + self.kl_beta * kl_divergence_error + self.triplet_beta * triplet_error

        loss_dict = {
            "loss": final_loss,
            "triplet_loss": triplet_error,
            "recon_loss": recon_error,
            "kl_divergence_loss": kl_divergence_error,
            "triplet_percentage": triplet_percentage
        }

        return loss_dict
