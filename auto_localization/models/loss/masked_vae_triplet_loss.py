import torch
from torch import nn
import torch.nn.functional as F

"""
    Masked VAE Triplet Loss module 
"""
class MaskedVAETripletLoss(nn.Module):
 
    def __init__(self, kl_beta = 1.0, triplet_beta = 1.0, triplet_margin = 1.0, latent_dim = 2, similarity_dim=1, reconstructive_dim=1, attributes=[2, 3], masks=False, bce=True, device="cuda", component_weighting=[1.0, 1.0], recon_beta=1.0):
        super(MaskedVAETripletLoss, self).__init__()
        self.kl_beta = kl_beta
        self.triplet_beta = triplet_beta
        self.recon_beta = recon_beta
        self.triplet_margin = triplet_margin
        self.latent_dim = latent_dim
        self.masks = masks
        self.train_mode = True
        self.attributes = attributes
        self.similarity_dim = similarity_dim
        self.reconstructive_dim = reconstructive_dim
        self.component_weighting = component_weighting
        self.bce = bce
        # define the triplet mask vectors 
        self.eps = 1e-7
        self.sigmoid = nn.Sigmoid()
        self.device = device
        self._setup_masks()

    def _setup_masks(self):
        self.masks = {}
        for index, attribute in enumerate(self.attributes):
            mask = torch.zeros(self.latent_dim)            
            mask[index] = self.component_weighting[index]
            self.masks[attribute] = mask

    def triplet_loss_masked(self, anchor, positive, negative, attribute_index=-1):
        # get the mask corresponding to the index
        if not isinstance(attribute_index, torch.Tensor) and attribute_index == -1:
            mask_vectors = [torch.ones(self.latent_dim)]
        else:
            mask_vectors = []
            for attribute_ind in attribute_index:
                mask_vectors.append(self.masks[int(attribute_ind)])

        mask_vectors = torch.stack(mask_vectors)
        mask_vectors = mask_vectors.to(self.device)
        mask_vectors = mask_vectors.unsqueeze(1)
        # apply the mask to each of the latent dimensions
        positive_anchor_distance = torch.abs(anchor - positive).unsqueeze(2)
        # positive_anchor = positive_anchor_distance * mask_vectors
        positive_anchor = torch.bmm(mask_vectors, positive_anchor_distance).squeeze()
        negative_anchor_distance = torch.abs(anchor - negative).unsqueeze(2)
        #negative_anchor = negative_anchor_distance * mask_vectors
        negative_anchor = torch.bmm(mask_vectors, negative_anchor_distance).squeeze()
        # get the differences of distances
        masked_distance_difference = positive_anchor - negative_anchor
        # calculate the final masked triplet loss
        masked_triplet_loss = F.relu(masked_distance_difference + self.triplet_margin)
        # get mean
        masked_triplet_loss = torch.mean(masked_triplet_loss)
        return masked_triplet_loss

    def triplet_loss_unmasked(self, anchor, positive, negative, attribute_index=-1):
        positive_anchor_distance = torch.abs(anchor - positive).unsqueeze(2)
        negative_anchor_distance = torch.abs(anchor - negative).unsqueeze(2)
        # get the differences of distances
        distance_difference = positive_anchor - negative_anchor
        # calculate the final masked triplet loss
        triplet_loss = F.relu(distance_difference + self.triplet_margin)
        # get mean
        triplet_loss = torch.mean(triplet_loss)
        return triplet_loss

    def triplet_percentage(self, anchor, positive, negative, attribute_index=-1):
        # get the mask corresponding to the index
        if not isinstance(attribute_index, torch.Tensor) and attribute_index == -1:
            mask_vectors = [torch.ones(self.latent_dim)]
        else:
            mask_vectors = []
            for attribute_ind in attribute_index:
                mask_vectors.append(self.masks[int(attribute_ind)])

        mask_vectors = torch.stack(mask_vectors)
        mask_vectors = mask_vectors.to(self.device)
        mask_vectors = mask_vectors.unsqueeze(1)
        # apply the mask to each of the latent dimensions
        positive_anchor_distance = torch.abs(anchor - positive).unsqueeze(2)
        #positive_anchor = mask_vectors * positive_anchor_distance
        positive_anchor = torch.bmm(mask_vectors, positive_anchor_distance).squeeze()
        negative_anchor_distance = torch.abs(anchor - negative).unsqueeze(2)
        negative_anchor = torch.bmm(mask_vectors, negative_anchor_distance).squeeze()
        #negative_anchor = mask_vectors * negative_anchor_distance
        # get the differences of distances
        masked_distance_difference = positive_anchor - negative_anchor
        # if it is negative then the triplet is passed
        is_negative = masked_distance_difference < 0
        if len(masked_distance_difference.shape) == 0:
            return 1.0 if is_negative else 0.0
        percentage = torch.sum(is_negative) / is_negative.shape[0]

        return percentage

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

    def forward(self, real_data, fake_data, mean, logvar, triplet_data, **kwargs):
        if len(triplet_data) == 4:
            anchor, positive, negative, attribute_index = triplet_data
        elif len(triplet_data) == 3:
            anchor, positive, negative = triplet_data
            attribute_index = -1
        anchor, positive, negative = anchor[0].to(self.device), positive[0].to(self.device), negative[0].to(self.device) # unpack means and ignore logvars
        """
        if len(anchor.shape) == 2:
            anchor = anchor.unsqueeze(0)
            anchor = positive.unsqueeze(0)
            anchor = negative.unsqueeze(0)
        """
        if self.masks:
            triplet_error = self.triplet_loss_masked(anchor, positive, negative, attribute_index)
        else:
            triplet_error = self.triplet_loss_unmasked(anchor, positive, negative, attribute_index)
        triplet_percentage = self.triplet_percentage(anchor, positive, negative, attribute_index)
        recon_error = self.reconstruction_loss(real_data, fake_data)
        kl_divergence_error = self.kl_divergence_loss(mean, logvar)

        final_loss = self.recon_beta * recon_error + self.kl_beta * kl_divergence_error + self.triplet_beta * triplet_error
        loss_dict = {
            "loss": final_loss,
            "triplet_loss": triplet_error,
            "recon_loss": recon_error,
            "kl_divergence_loss": kl_divergence_error,
            "triplet_percentage": triplet_percentage
        }

        return loss_dict

 
