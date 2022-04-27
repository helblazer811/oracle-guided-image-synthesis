import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

"""
    Global Mask VAE Triplet Loss 
"""
class GlobalMaskVAETripletLoss(nn.Module):

    def __init__(self, kl_beta = 1.0, triplet_beta = 1.0, mask_regularization_beta = 0.0, triplet_margin = 1.0, latent_dim = 2):
        super(GlobalMaskVAETripletLoss, self).__init__()
        self.kl_beta = kl_beta
        self.triplet_beta = triplet_beta
        self.triplet_margin = triplet_margin
        if mask_regularization_beta is None:
            mask_regularization_beta = 0.0
        self.mask_regularization_beta = mask_regularization_beta
        self.latent_dim = latent_dim
        # define the global mask vector 
        self.mask_vector = nn.Linear(1, self.latent_dim)
        #self.mask_activation = nn.Sigmoid()
        # self.mask_activation = nn.ReLU()
        self.mask_activation = nn.Softmax()
        self.device = "cuda"

    def triplet_loss(self, anchor, positive, negative, exclude_dim=None, **kwargs):
        # calculate sigmoid mask vector 
        mask_vector_tensor = self.mask_vector(torch.Tensor([1]).to(self.device))# self.mask_vector.weight)#.to(self.device) 
        mask_vector = self.mask_activation(mask_vector_tensor)
        if not exclude_dim is None:
            dim_range = np.arange(0, self.latent_dim)
            if exclude_dim != -1:
                dim_range = np.ma.array(dim_range, mask=False)
                dim_range.mask[exclude_dim] = True    
                dim_range = dim_range.compressed()
            dim_range = torch.LongTensor(dim_range).to(self.device)
            mask_vector = mask_vector.index_select(0, dim_range) 
        # multiply the mask vector by each input vector
        masked_anchor = torch.mul(mask_vector, anchor)
        masked_positive = torch.mul(mask_vector, positive)
        masked_negative = torch.mul(mask_vector, negative)
        # calculate the triplet loss
        triplet_loss = F.triplet_margin_loss(masked_anchor, masked_positive, masked_negative, margin=self.triplet_margin)
        
        return triplet_loss

    def reconstruction_loss(self, real_data, fake_data):
        rec_loss = F.binary_cross_entropy(fake_data, real_data, reduction='mean')
        return rec_loss

    def kl_divergence_loss(self, mean, logvar):
        kl_div = -.5 * (1. + logvar - mean ** 2 - logvar.exp()).mean()
        return kl_div

    def get_mask_activation_vector(self):
        mask_vector_tensor = self.mask_vector(torch.Tensor([1]).to(self.device))
        mask_vector = self.mask_activation(mask_vector_tensor)
        return mask_vector
    
    def mask_regularization_loss(self):
        # compute l1 norm of mask vector
        mask_vector_tensor = self.mask_vector(torch.Tensor([1]).to(self.device))
        mask_vector = self.mask_activation(mask_vector_tensor)
        loss = torch.norm(mask_vector, p=1)
        return loss

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
        mask_regularization_loss = self.mask_regularization_loss()

        final_loss = recon_error + self.kl_beta * kl_divergence_error + self.triplet_beta * triplet_error + self.mask_regularization_beta * mask_regularization_loss
        return final_loss, triplet_error, recon_error, kl_divergence_error

 
