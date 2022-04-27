import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
import numpy as np
import wandb

"""
    Cycle Consistent Triplet Loss
"""
class CycleConsistentTripletLoss(nn.Module):

    def __init__(self, kl_beta = 1.0, triplet_beta = 1.0, triplet_margin = 1.0, latent_dim=2, similarity_dim=2, reconstructive_dim=2, bce=False, recon_beta=1.0, cycle_consistent_beta=1.0, similarity_kl_beta=1.0, reparameterize_triplet=False, logistic_triplet=False, pairwise_cycle_consistency=False, logistic_triplet_squared=False, **kwargs):
        super(CycleConsistentTripletLoss, self).__init__()
        self.device = "cuda"
        self.kl_beta = kl_beta
        self.similarity_kl_beta = similarity_kl_beta
        self.triplet_beta = triplet_beta
        self.recon_beta = recon_beta
        self.cycle_consistent_beta = cycle_consistent_beta
        self.logistic_triplet = logistic_triplet
        self.triplet_margin = triplet_margin
        self.latent_dim = latent_dim
        self.similarity_dim = similarity_dim
        self.pairwise_cycle_consistency = pairwise_cycle_consistency
        self.reconstructive_dim = reconstructive_dim
        self.logistic_triplet_squared = logistic_triplet_squared
        self.reparameterize_triplet = reparameterize_triplet
        # dummy layer to appease nn module requirements
        self.empty_linear = nn.Linear(1, 1)
        self.triplet_margin_loss = nn.TripletMarginLoss(margin=self.triplet_margin, p=2)
        self.learnable_k = nn.Parameter(torch.Tensor([1.0]))
        self.learnable_k.requires_grad = True
        self.bce = bce
    
    """
        Measures triplet loss
    """
    def triplet_loss(self, anchor, positive, negative, **kwargs):
        if self.logistic_triplet and not self.logistic_triplet_squared:
            distance_anchor_positive = torch.norm(anchor - positive, dim=-1)
            distance_anchor_negative = torch.norm(anchor - negative, dim=-1)

            def logistic_function(x):
                return 1 / (1 + torch.exp(-1*x))
             
            wandb.log({"k_val": self.learnable_k.data[0]})
            loss = -1*torch.log(logistic_function(self.learnable_k * (distance_anchor_negative**2 - distance_anchor_positive**2)))
            return loss.mean()
        elif self.logistic_triplet_squared:
            
            def distance_function(a, b):
                return torch.norm(a - b, dim=-1)**2 

            loss = F.triplet_margin_with_distance_loss(anchor, positive, negative, distance_function=distance_function, margin=self.triplet_margin)
            return loss

        else:
            return self.triplet_margin_loss(anchor, positive, negative)
  
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

    """
        Evaluates the cycle consistency loss. 
            - Embeds an image
    """
    def cycle_consistent_loss(self, model, real_data):
        
        def similarity_cycle():
            mu, logvar = model.encode(real_data)
            # break up mu into similarity and other
            mu_similarity = mu[:, :self.similarity_dim]
            mu_reconstructive = mu[:, self.similarity_dim:]
            # keep the old similarity dim
            z_similarity = mu_similarity
            # sample the random reconstructive dim from the prior 
            noise = torch.normal(0.0, 1.0, size=mu_reconstructive.shape).cuda() 
            z_reconstructive = noise
            z_concat = torch.cat((z_similarity, z_reconstructive), axis=1)
            # decode the vector
            x_pred = model.decode(z_concat)
            # encode the given image
            mu_new, logvar_new = model.encode(x_pred)
            # get the new similarity vector
            mu_new_similarity = mu_new[:, :self.similarity_dim]
            # compute the l2 norm between the original similarity and the new similarity 
            distance = torch.norm(mu_new_similarity - mu_similarity, dim=1)
            
            return distance.mean()

        def reconstructive_cycle():
            mu, logvar = model.encode(real_data)
            # break up mu into similarity and other
            mu_similarity = mu[:, :self.similarity_dim]
            mu_reconstructive = mu[:, self.similarity_dim:]
            # keep the old reconstructive dim
            z_reconstructive = mu_reconstructive
            # sample the random reconstructive dim from the prior 
            noise = torch.normal(0.0, 1.0, size=mu_similarity.shape).cuda() 
            z_similarity = noise
            z_concat = torch.cat((z_similarity, z_reconstructive), axis=1)
            # decode the vector
            x_pred = model.decode(z_concat)
            # encode the given image
            mu_new, logvar_new = model.encode(x_pred)
            # get the new similarity vector
            mu_new_recon = mu_new[:, self.similarity_dim:]
            # compute the l2 norm between the original similarity and the new similarity 
            distance = torch.norm(mu_new_recon - mu_reconstructive, dim=1)
        
            return distance.mean()

        similarity_cycle_loss = similarity_cycle()        
        reconstructive_cycle_loss = reconstructive_cycle()

        return similarity_cycle_loss + reconstructive_cycle_loss

    def pairwise_cycle_consistent_loss(self, model, real_data):
        mu, logvar = model.encode(real_data)
        # break up mu into similarity and other
        mu_similarity = mu[:, :self.similarity_dim]
        mu_reconstructive = mu[:, self.similarity_dim:]
        def similarity_cycle():
            # keep the old similarity dim
            z_similarity = mu_similarity
            # sample the random reconstructive dim from the prior 
            noise = torch.normal(0.0, 1.0, size=mu_reconstructive.shape).cuda() 
            z_reconstructive = noise
            z_concat = torch.cat((z_similarity, z_reconstructive), axis=1)
            # decode the vector
            x_pred = model.decode(z_concat)
            # encode the given image
            mu_new, logvar_new = model.encode(x_pred)
            # get the new similarity vector
            mu_new_similarity = mu_new[:, :self.similarity_dim]
            # compute the l2 norm between the original similarity and the new similarity 
            distance = torch.norm(mu_new_similarity - mu_similarity, dim=1)
            
            return distance.mean()

        def reconstructive_cycle():
            similarity_a = mu_similarity
            similarity_b = mu_similarity[:][torch.randperm(real_data.shape[0])]
            # decode a 
            a_concat = torch.cat((similarity_a, mu_reconstructive), axis=1)
            a_image = model.decode(a_concat)
            # decode b
            b_concat = torch.cat((similarity_b, mu_reconstructive), axis=1)
            b_image = model.decode(b_concat)
            # encdoe the images
            mu_a, _ = model.encode(a_image)
            reconstructive_a = mu_a[:, self.similarity_dim:]
            mu_b, _ = model.encode(b_image)
            reconstructive_b = mu_b[:, self.similarity_dim:]
            # compute the l2 norm between the original similarity and the new similarity 
            distance = torch.norm(reconstructive_a - reconstructive_b, dim=1)
        
            return distance.mean()

        # don't use pairwise loss for the similarity cycle
        # this is because we want the similarity space to represent a unique embedding
        similarity_cycle_loss = similarity_cycle()        
        reconstructive_cycle_loss = reconstructive_cycle()

        return similarity_cycle_loss + reconstructive_cycle_loss

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    """
        This is the autoencoder specific loss that is passed to the entire network
    """
    def auto_encoder_loss(self, real_data, fake_data, mean, logvar):
        recon_error = self.reconstruction_loss(real_data, fake_data)
        kl_divergence_error = self.kl_divergence_loss(mean, logvar)
        combined_loss = self.recon_beta * recon_error + self.kl_beta * kl_divergence_error

        loss_dict = {
            "loss": combined_loss,
            "recon_loss": recon_error,
            "kl_divergence_loss": kl_divergence_error,
        }

        return loss_dict

    """
        This is the reverse cycle  loss that is passed specifically to the encoder
    """
    def reverse_cycle_loss(self, real_data, fake_data, mean, logvar, triplet_data, **kwargs):
        if "model" in kwargs:
            model = kwargs["model"]
            if self.pairwise_cycle_consistency:
                cycle_consistent_error = self.pairwise_cycle_consistent_loss(model, real_data)
            else:
                cycle_consistent_error = self.cycle_consistent_loss(model, real_data)
          
        if len(triplet_data) == 4:
            anchor, positive, negative, attribute_index = triplet_data
        else:
            anchor, positive, negative = triplet_data
        anchor_mean, anchor_logvar = anchor
        positive_mean, positive_logvar = positive
        negative_mean, negative_logvar = negative
        if self.reparameterize_triplet: 
            anchor = self.reparameterize(anchor_mean, anchor_logvar)
            positive = self.reparameterize(positive_mean, positive_logvar)
            negative = self.reparameterize(negative_mean, negative_logvar)
        else:
            anchor = anchor_mean
            positive = positive_mean
            negative = negative_mean

        triplet_error = self.triplet_loss(anchor, positive, negative)
        triplet_percentage = self.triplet_percentage(anchor, positive, negative)
        final_loss = self.triplet_beta * triplet_error + self.cycle_consistent_beta * cycle_consistent_error

        loss_dict = {
            "loss": final_loss,
            "triplet_loss": triplet_error,
            "triplet_percentage": triplet_percentage,
            "cycle_consistent_loss": cycle_consistent_error
        }

        return loss_dict

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
        anchor_mean, anchor_logvar = anchor
        positive_mean, positive_logvar = positive
        negative_mean, negative_logvar = negative
        if self.reparameterize_triplet: 
            anchor = self.reparameterize(anchor_mean, anchor_logvar)
            positive = self.reparameterize(positive_mean, positive_logvar)
            negative = self.reparameterize(negative_mean, negative_logvar)
        else:
            anchor = anchor_mean
            positive = positive_mean
            negative = negative_mean
        # segment the triplet
        anchor = anchor[:, :self.similarity_dim]
        positive = positive[:, :self.similarity_dim]
        negative = negative[:, :self.similarity_dim]
        if test_mode or not self.triplet_beta == 0.0:
            triplet_error = self.triplet_loss(anchor, positive, negative)
        else:
            triplet_error = torch.Tensor([0.0]).to(self.device)

        if "model" in kwargs:
            model = kwargs["model"]
            if self.pairwise_cycle_consistency:
                cycle_consistent_error = self.pairwise_cycle_consistent_loss(model, real_data)
            else:
                cycle_consistent_error = self.cycle_consistent_loss(model, real_data)

        recon_error = self.reconstruction_loss(real_data, fake_data)
        triplet_percentage = self.triplet_percentage(anchor, positive, negative)
        # split up the mean and logvar
        similarity_mean = mean[:, :self.similarity_dim]
        similarity_logvar = logvar[:, :self.similarity_dim]
        reconstructive_mean = mean[:, self.similarity_dim:]
        reconstructive_logvar = logvar[:, self.similarity_dim:]

        """
        similarity_kl_divergence_error = self.kl_divergence_loss(similarity_mean, similarity_logvar)
        if self.latent_dim - self.similarity_dim > 0:
            reconstructive_kl_divergence_error = self.kl_divergence_loss(reconstructive_mean, reconstructive_logvar)
        else:
            reconstructive_kl_divergence_error = torch.Tensor([0.0]).cuda()
        kl_divergence_error = similarity_kl_divergence_error + reconstructive_kl_divergence_error
        """
        kl_divergence_error = self.kl_divergence_loss(mean, logvar)
        # kl divergence is prescaled
        """
        final_loss = self.recon_beta * recon_error + \
                    self.kl_beta * reconstructive_kl_divergence_error + \
                    self.similarity_kl_beta * similarity_kl_divergence_error + \
                    self.triplet_beta * triplet_error + \
                    self.cycle_consistent_beta * cycle_consistent_error
        """
        final_loss = self.recon_beta * recon_error + \
                    self.kl_beta * kl_divergence_error + \
                    self.triplet_beta * triplet_error + \
                    self.cycle_consistent_beta * cycle_consistent_error

        loss_dict = {
            "loss": final_loss,
            "triplet_loss": triplet_error,
            "recon_loss": recon_error,
            "kl_divergence_loss": kl_divergence_error,
#            "similarity_kl_divergence": similarity_kl_divergence_error,
#            "reconstructive_kl_divergence": reconstructive_kl_divergence_error,
            "triplet_percentage": triplet_percentage,
            "cycle_consistent_loss": cycle_consistent_error
        }

        return loss_dict
