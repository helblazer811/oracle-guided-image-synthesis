import torch
from torch import nn
import torch.nn.functional as F

"""
    VAE Triplet Loss module 
"""
class VAELoss(nn.Module):

    def __init__(self, kl_beta = 1.0):
        super(VAELoss, self).__init__()
        self.kl_beta = kl_beta
        # empty linear to appease nn module requirement
        self.empty_linear = nn.Linear(1, 1)
        self.device = "cuda"

    def reconstruction_loss(self, real_data, fake_data):
        rec_loss = F.mse_loss(fake_data, real_data, reduction="mean")
        return rec_loss

    def kl_divergence_loss(self, mean, logvar):
        kl_div = -.5 * (1. + logvar - mean ** 2 - logvar.exp()).mean()
        return kl_div

    def forward(self, real_data, fake_data, mean, logvar, triplet_data, **kwargs):
        test_mode = False
        if "test_mode" in kwargs:
            test_mode = kwargs["test_mode"]
        # latent vectors of the anchor, positive, and negative data
        anchor, positive, negative = triplet_data
        recon_error = self.reconstruction_loss(real_data, fake_data)
        kl_divergence_error = self.kl_divergence_loss(mean, logvar)
        triplet_error = torch.Tensor([0.0])

        final_loss = recon_error + self.kl_beta * kl_divergence_error
        loss_dict = {
            "loss": final_loss,
            "triplet_loss": triplet_error,
            "recon_loss": recon_error,
            "kl_divergence_loss": kl_divergence_error,
        }

        return loss_dict

 
