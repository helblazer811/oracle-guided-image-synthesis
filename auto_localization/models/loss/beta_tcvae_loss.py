import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')
import math

class BetaTCVAELoss(nn.Module):
    num_iter = 0 # Global static variable to keep track of iterations
    
    def __init__(self, gamma=0.0, kl_beta=0.0, dataset_size=0, anneal_steps=10000, alpha=1):
        super(BetaTCVAELoss, self).__init__()
        self.empty_linear = nn.Linear(1, 1)
        self.dataset_size = dataset_size
        self.anneal_steps = anneal_steps
        self.alpha = alpha
        self.num_iter = 0
        self.gamma = 0
        self.kl_beta = 0

    def log_density_gaussian(self, x: Tensor, mu: Tensor, logvar: Tensor):
        """
        Computes the log pdf of the Gaussian with parameters mu and logvar at x
        :param x: (Tensor) Point at whichGaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param logvar: (Tensor) Log variance of the Gaussian distribution
        :return:
        """
        norm = - 0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, real_data, fake_data, mean, log_var, triplet_data, **kwargs):
        test_mode = False
        if "test_mode" in kwargs:
            test_mode = kwargs["test_mode"]
        # latent vectors of the anchor, positive, and negative data
        anchor, positive, negative, attribute_index = triplet_data
        z = self.reparameterize(mean, log_var)
        weight = 1 #kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(fake_data, real_data, reduction='sum')
        log_q_zx = self.log_density_gaussian(z, mean, log_var).sum(dim = 1)

        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim = 1)

        batch_size, latent_dim = z.shape
        mat_log_q_z = self.log_density_gaussian(z.view(batch_size, 1, latent_dim),
                                                mean.view(1, batch_size, latent_dim),
                                                log_var.view(1, batch_size, latent_dim))

        # Reference
        # [1] https://github.com/YannDubs/disentangling-vae/blob/535bbd2e9aeb5a200663a4f82f1d34e084c4ba8d/disvae/utils/math.py#L54
        strat_weight = (self.dataset_size - batch_size + 1) / (self.dataset_size * (batch_size - 1))
        importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size -1)).to(real_data.device)
        importance_weights.view(-1)[::batch_size] = 1 / self.dataset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()

        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)

        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)

        mi_loss  = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if not test_mode:
            self.num_iter += 1
            anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
        else:
            anneal_rate = 1.

        loss = recons_loss / batch_size + \
               self.alpha * mi_loss + \
               weight * (self.kl_beta * tc_loss +
                         anneal_rate * self.gamma * kld_loss)
        
        return {
            'loss': loss,
            'recon_loss':recons_loss,
            'kl_divergence_loss':kld_loss,
            'triplet_loss': torch.Tensor([0.0]),
            'TC_Loss':tc_loss,
            'MI_Loss':mi_loss
        }

