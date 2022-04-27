import torch
from torch import nn
import torch.nn.functional as F

"""
    Logistic Triplet Loss module 
"""
class LogisticTripletLoss(nn.Module):

    def __init__(self, kl_beta = 1.0, triplet_beta = 1.0):
        super(LogisticTripletLoss, self).__init__()
        self.kl_beta = kl_beta
        self.triplet_beta = triplet_beta
        # empty linear to appease nn module requirement
        self.empty_linear = nn.Linear(1, 1)
        self.device = "cuda"

    def triplet_loss(self, anchor, positive, negative, **kwargs):
        # logistic triplet loss
        loss = torch.sigmoid(torch.norm(anchor - positive, dim=1)**2 - torch.norm(anchor - negative, dim=1)**2)
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
            anchor, positive, negative = anchor[0], positive[0], negative[0] # unpack means and ignore logvars
            triplet_error = self.triplet_loss(anchor, positive, negative)
        else:
            triplet_error = torch.Tensor([0.0]).to(self.device)
        recon_error = self.reconstruction_loss(real_data, fake_data)
        kl_divergence_error = self.kl_divergence_loss(mean, logvar)

        final_loss = recon_error + self.kl_beta * kl_divergence_error + self.triplet_beta * triplet_error
        return final_loss, triplet_error, recon_error, kl_divergence_error

 
