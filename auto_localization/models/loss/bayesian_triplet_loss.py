import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import wandb

def get_ll_mean_variance(muA, muP, muN, varA, varP, varN, margin = 0.0):
    muA2 = muA**2
    muP2 = muP**2
    muN2 = muN**2
    varP2 = varP**2
    varN2 = varN**2

    mu = torch.sum(muP2 + varP - muN2 - varN - 2*muA*(muP - muN), dim=1)
    T1 = varP2 + 2*muP2 * varP + 2*(varA + muA2)*(varP + muP2) - 2*muA2 * muP2 - 4*muA*muP*varP
    T2 = varN2 + 2*muN2 * varN + 2*(varA + muA2)*(varN + muN2) - 2*muA2 * muN2 - 4*muA*muN*varN
    T3 = 4*muP*muN*varA
    sigma2 = torch.sum(2*T1 + 2*T2 - 2*T3, dim=1)

    return sigma2, mu

def negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin = 0.0):
    muA2 = muA**2
    muP2 = muP**2
    muN2 = muN**2
    varP2 = varP**2
    varN2 = varN**2

    mu = torch.sum(muP2 + varP - muN2 - varN - 2*muA*(muP - muN), dim=1)
    #wandb.log({"mu": torch.mean(mu)})
    T1 = varP2 + 2*muP2 * varP + 2*(varA + muA2)*(varP + muP2) - 2*muA2 * muP2 - 4*muA*muP*varP
    T2 = varN2 + 2*muN2 * varN + 2*(varA + muA2)*(varN + muN2) - 2*muA2 * muN2 - 4*muA*muN*varN
    T3 = 4*muP*muN*varA
    sigma2 = torch.sum(2*T1 + 2*T2 - 2*T3, dim=1)
    sigma = sigma2**0.5

    #wandb.log({"sigma": torch.mean(sigma)})
    probs = Normal(loc = mu, scale = sigma + 1e-8).cdf(-1*margin)
    #wandb.log({"probs": torch.mean(probs)})
    nll = -torch.log(probs + 1e-8)

    return nll.mean()

def kl_div_gauss(mu_q, var_q, mu_p, var_p):
    N, D = mu_q.shape
    # kl diverence for isotropic gaussian
    kl = 0.5 * ((var_q / var_p) * D + 1.0 / (var_p) * torch.sum(mu_p**2 + \
        mu_q**2 - 2*mu_p*mu_q, axis=1) - \
        D + \
        D*(torch.log(var_p) - torch.log(var_q)))

    return kl.mean()

def kl_divergence_unit_gauss(mean, var):
    logvar = torch.log(var)
    kl_div = -.5 * (1. + logvar - mean ** 2 - logvar.exp()).mean()
    return kl_div

def kl_div_vMF(mu_q, var_q):
    N, D = mu_q.shape

    # we are estimating the variance and not kappa in the network.
    # They are propertional
    kappa_q = 1.0 / var_q
    kl = kappa_q - D * torch.log(2.0)

    return kl.mean()

class BayesianTripletLoss(nn.Module):

    def __init__(self, triplet_margin=1.0, kl_beta=1.0, triplet_beta=1.0, latent_dim=2, distribution='gauss', varPrior=1.0, similarity_dim=2, attributes=[2, 3], masks=False, recon_beta=1.0, bce=True, isolated_warmup=0, reconstructive_dim=0, reparameterize_triplet=False, uncertainty_constant=False, sub_loss_type=False):
        super(BayesianTripletLoss, self).__init__()
        self.epoch = 0
        self.latent_dim = latent_dim
        self.similarity_dim = similarity_dim
        self.reconstructive_dim = reconstructive_dim
        self.recon_beta = recon_beta
        self.reparameterize_triplet = reparameterize_triplet
        self.triplet_margin = triplet_margin
        self.varPrior = varPrior
        self.attributes = attributes
        self.kl_beta = kl_beta
        self.masks = masks
        self.sub_loss_type = sub_loss_type
        self.triplet_beta = triplet_beta
        self.uncertainty_constant = uncertainty_constant
        self.distribution = distribution
        self.device = "cuda"
        self.bce = bce
        self.isolated_warmup = isolated_warmup
        self.triplet_margin = triplet_margin
        self.empty_linear = nn.Linear(1, 1)
        self.triplet_margin_loss = nn.TripletMarginLoss(margin=self.triplet_margin)
        self._setup_masks()

    def _setup_masks(self):
        self.masks = {}
        for index, attribute in enumerate(self.attributes):
            mask = torch.zeros(self.latent_dim)            
            mask[index] = 1.0
            self.masks[attribute] = mask

    def bayesian_triplet_loss_unmasked(self, triplet_data):
        anchor, positive, negative, attribute_index = triplet_data
        # apply the mask to each of the latent dimensions
        varA = anchor[1].exp()
        varP = positive[1].exp()
        varN = negative[1].exp()
        #wandb.log({"mean variance": torch.mean(torch.cat((varA, varP, varN)))})
        muA = anchor[0]
        muP = positive[0]
        muN = negative[0]

        negative_log_likelihood_loss = negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin=self.triplet_margin)

        return negative_log_likelihood_loss

    def bayesian_triplet_loss_masked(self, triplet_data):
        anchor, positive, negative, attribute_index = triplet_data
        # get the mask corresponding to the index
        if not isinstance(attribute_index, torch.Tensor) and attribute_index == -1:
            mask_vectors = [torch.ones(self.latent_dim)]
        else:
            mask_vectors = []
            for attribute_ind in attribute_index:
                mask_vectors.append(self.masks[int(attribute_ind)])

        mask_vectors = torch.stack(mask_vectors)
        mask_vectors = mask_vectors.to(self.device)
        # mask_vectors = mask_vectors.unsqueeze(1)
        mask_vectors = mask_vectors.bool()
        # apply the mask to each of the latent dimensions
        varA = anchor[1].exp()
        varP = positive[1].exp()
        varN = negative[1].exp()
        #wandb.log({"mean variance": torch.mean(torch.cat((varA, varP, varN)))})
        muA = anchor[0]
        muP = positive[0]
        muN = negative[0]
        # num masked per row = 1
        num_masked_per_row = 1
        varA = torch.masked_select(varA, mask_vectors).reshape(-1, num_masked_per_row)
        varP = torch.masked_select(varP, mask_vectors).reshape(-1, num_masked_per_row)
        varN = torch.masked_select(varN, mask_vectors).reshape(-1, num_masked_per_row)

        muA = torch.masked_select(muA, mask_vectors).reshape(-1, num_masked_per_row)
        muP = torch.masked_select(muP, mask_vectors).reshape(-1, num_masked_per_row)
        muN = torch.masked_select(muN, mask_vectors).reshape(-1, num_masked_per_row)

        negative_log_likelihood_loss = negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin=self.triplet_margin)

        return negative_log_likelihood_loss

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
        if len(triplet_data) == 4:
            anchor, positive, negative, attribute_index = triplet_data
        else:
            anchor, positive, negative = triplet_data
        anchor_mean, anchor_logvar = anchor
        positive_mean, positive_logvar = positive
        negative_mean, negative_logvar = negative
        """
        if self.reparameterize_triplet: 
            anchor = self.reparameterize(anchor_mean, anchor_logvar)
            positive = self.reparameterize(positive_mean, positive_logvar)
            negative = self.reparameterize(negative_mean, negative_logvar)
        else:
            anchor = anchor_mean
            positive = positive_mean
            negative = negative_mean
        """
        if self.sub_loss_type == "combined":
            triplet_error = self.combined_triplet_loss(triplet_data)
        elif self.sub_loss_type == "isolated":
            triplet_error = self.isolated_triplet_loss(triplet_data)
        elif self.sub_loss_type == "bayesian":
            triplet_error = self.bayesian_triplet_loss_unmasked(triplet_data)
        else:
            raise Exception(f"Unrecognized loss type : {self.sub_loss_type}")

        triplet_percentage = self.triplet_percentage(triplet_data)
        final_loss = self.triplet_beta * triplet_error

        loss_dict = {
            "loss": final_loss,
            "triplet_loss": triplet_error,
            "cycle_consistent_loss": triplet_error,
            "triplet_percentage": triplet_percentage
        }

        return loss_dict

    def isolated_triplet_loss(self, triplet_data, **kwargs):
        anchor, positive, negative, attribute_index = triplet_data
        muA = anchor[0]
        muP = positive[0]
        muN = negative[0]

        return F.triplet_margin_loss(muA, muP, muN, margin=self.triplet_margin)

    def kl_divergence_loss(self, mean, logvar):
        kl_div = -.5 * (1. + logvar - mean ** 2 - logvar.exp()).mean()
        return kl_div

    """
        Computes the kl divergence of the triplet
    """
    def kl_divergence_triplet(self, triplet_data):
        anchor, positive, negative, attribute_index = triplet_data

        varA = anchor[1].exp()
        varP = positive[1].exp()
        varN = negative[1].exp()
        muA = anchor[0]
        muP = positive[0]
        muN = negative[0]
        # KL(anchor|| prior) + KL(positive|| prior) + KL(negative|| prior)
        if self.distribution == 'gauss':
            muPrior = torch.zeros_like(muA, requires_grad = False)
            varPrior = torch.ones_like(varA, requires_grad = False) * self.varPrior
            kl = kl_divergence_unit_gauss(muA, varA) + \
                kl_divergence_unit_gauss(muP, varP) + \
                kl_divergence_unit_gauss(muN, varN)
            #kl = (kl_div_gauss(muA, varA, muPrior, varPrior) + \
            #kl_div_gauss(muP, varP, muPrior, varPrior) + \
            #kl_div_gauss(muN, varN, muPrior, varPrior))
        elif self.distribution == 'vMF':
            kl = (kl_div_vMF(muA, varA) + \
            kl_div_vMF(muP, varP) + \
            kl_div_vMF(muN, varN))

        return kl

    """
        Computes the general reconstruction loss
    """
    def reconstruction_loss(self, real_data, fake_data):
        if self.bce:
            rec_loss = F.binary_cross_entropy(fake_data, real_data, reduction='mean')
            return rec_loss
        else:
            rec_loss = F.mse_loss(fake_data, real_data, reduction="mean")
            return rec_loss

    def triplet_percentage(self, triplet_data):
        anchor, positive, negative, attribute_index = triplet_data
        anchor = anchor[0]
        positive = positive[0]
        negative = negative[0]
        # calculate distances
        distance_anchor_positive = torch.norm(anchor - positive, dim=-1)
        distance_anchor_negative = torch.norm(anchor - negative, dim=-1)
        # test if it is negative
        num_closer = torch.sum((distance_anchor_positive < distance_anchor_negative).int())
        percentage = torch.Tensor([num_closer/anchor.shape[0]])
        return percentage

    """
        Alias for bayesian_triplet_loss_unmasked
    """
    def triplet_loss(self, triplet_data):
        return self.bayesian_triplet_loss_unmasked(triplet_data)

    def combined_triplet_loss(self, triplet_data):
        btl = 2*self.bayesian_triplet_loss_unmasked(triplet_data)
        isolated = self.isolated_triplet_loss(triplet_data)
        combined = (btl + isolated)/2
        return combined

    def forward(self, real_data, fake_data, mean, logvar, triplet_data, **kwargs):
        # divide x into anchor, positive, negative based on labels
        """
        D, N = x.shape
        nq = torch.sum(label.data == -1).item() # number of tuples
        S = x.size(1) // nq # number of images per tuple including query: 1+1+n
        A = x[:, label.data == -1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, D).permute(1, 0)
        P = x[:, label.data == 1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, D).permute(1, 0)
        N = x[:, label.data == 0]
        """
        kl_divergence_error = self.kl_divergence_loss(mean, logvar) # kl divergence of the real/fake data
        # kl_divergence_triplet_error = self.kl_divergence_triplet(triplet_data) # kl_divergence of the triplets

        if self.sub_loss_type == "combined":
            triplet_error = self.combined_triplet_loss(triplet_data)
        elif self.sub_loss_type == "isolated":
            triplet_error = self.isolated_triplet_loss(triplet_data)
        elif self.sub_loss_type == "bayesian":
            if self.masks: # use masks or don't
                triplet_error = self.bayesian_triplet_loss_masked(triplet_data)
            else:
                triplet_error = self.bayesian_triplet_loss_unmasked(triplet_data)
        else:
            raise Exception(f"Unrecognized loss type : {self.sub_loss_type}")

        recon_error = self.reconstruction_loss(real_data, fake_data)
    
        triplet_percentage = self.triplet_percentage(triplet_data)

        final_loss = self.recon_beta * recon_error + \
                    self.kl_beta * kl_divergence_error + \
                    self.triplet_beta * triplet_error

        loss_dict = {
            "loss": final_loss,
            "triplet_loss": triplet_error,
            "recon_loss": recon_error,
            "kl_divergence_loss": kl_divergence_error,
        #    "kl_divergence_triplet_loss": kl_divergence_triplet_error
            "triplet_percentage": triplet_percentage
        }

        return loss_dict
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.triplet_margin) + ')'
