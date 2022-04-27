import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import util
from loss.masked_vae_triplet_loss import MaskedVAETripletLoss

class MaskedVAEIsolated(nn.Module):

    def __init__(self, latent_dim, in_shape, similarity_dim=1, reconstructive_dim=1, layer_count=4, channels=1, d=128, kl_beta=1.0, triplet_beta=1.0, triplet_margin=1.0, loss_name=None):
        super(MaskedVAEIsolated, self).__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.z_dim = self.latent_dim
        self.similarity_dim = similarity_dim
        self.reconstructive_dim = reconstructive_dim
        if not self.latent_dim == (self.similarity_dim + self.reconstructive_dim):
            raise Exception("Similarity dim and Reconstructive dim do not sum to latent dim")
        self.similarity_mode = False
        self.in_shape = in_shape
        self.layer_count = layer_count
        self.channels = channels
        self.kl_beta = kl_beta
        self.triplet_beta = triplet_beta
        self.triplet_margin = triplet_margin
        self.loss_name = loss_name
        # run model setup
        self.loss_function = MaskedVAETripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta, triplet_margin=self.triplet_margin, latent_dim=self.latent_dim, similarity_dim=self.similarity_dim, reconstructive_dim=self.reconstructive_dim)
        self._setup_model()
   
    @classmethod
    def from_config(cls, config):
        return cls(
                latent_dim = config["latent_dim"],
                similarity_dim = config["similarity_dim"],
                reconstructive_dim = config["reconstructive_dim"],
                in_shape = config["in_shape"],
                d = config["d"],
                layer_count = config["layer_count"],
                channels = config["channels"],
                kl_beta = config["kl_beta"],
                triplet_beta = config["triplet_beta"],
                triplet_margin = config["triplet_margin"],
                loss_name = config["loss_name"],
        )

    def _setup_model(self):
        mul = 1
        inputs = self.channels
        out_sizes = [(self.in_shape, self.in_shape)]
        for i in range(self.layer_count):
            setattr(self, "conv%d" % (i + 1), nn.Conv2d(inputs, self.d * mul, 4, 2, 1))
            setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm2d(self.d * mul))
            h_w = (out_sizes[-1][-1], out_sizes[-1][-1])
            out_sizes.append(util.conv2d_output_shape(h_w, kernel_size=4, stride=2, pad=1, dilation=1))
            inputs = self.d * mul
            mul *= 2

        self.d_max = inputs
        self.last_size = out_sizes[-1][-1]
        self.num_linear = self.last_size ** 2 * self.d_max
        # isolate the linear layers for the recontructive embedding and similarity embedding
        self.similarity_mean_linear = nn.Linear(self.num_linear, self.similarity_dim)
        self.similarity_logvar_linear = nn.Linear(self.num_linear, self.similarity_dim)
        self.reconstructive_mean_linear = nn.Linear(self.num_linear, self.reconstructive_dim)
        self.reconstructive_logvar_linear = nn.Linear(self.num_linear, self.reconstructive_dim)

        self.d1 = nn.Linear(self.latent_dim, self.num_linear)

        mul = inputs // self.d // 2

        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, self.d * mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(self.d * mul))
            inputs = self.d * mul
            mul //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, self.channels, 4, 2, 1))
    
    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def encode(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        for i in range(self.layer_count):
            x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))
        x = x.view(x.shape[0], self.num_linear)
        # use encoding linear layers
        sim_mean = self.similarity_mean_linear(x)
        sim_logvar  = self.similarity_logvar_linear(x)
        if self.similarity_mode:
            return sim_mean, sim_logvar
        recon_mean = self.reconstructive_mean_linear(x)
        recon_logvar  = self.reconstructive_logvar_linear(x)
        # concat the outputs 
        mean = torch.cat((sim_mean, recon_mean), -1)
        logvar = torch.cat((sim_logvar, recon_logvar), -1)

        return mean, logvar
    
    """
        Utility function to break up embedding values
        into similarity and reconstructive subspaces
    """
    def segment_embedding(self, mean, logvar):
        sim_mean, recon_mean = torch.split(mean, [self.similarity_dim, self.reconstructive_dim], dim=-1)
        sim_logvar, recon_logvar = torch.split(logvar, [self.similarity_dim, self.reconstructive_dim], dim=-1)

        return (sim_mean, recon_mean), (sim_logvar, recon_logvar)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        if self.similarity_mode:
            # the input will only be a vector of size similarity
            # generate a random vector from the prior distribution to fill the rest of the vector
            # assume prior to be normal at this point
            batch_size = x.shape[0]
            if len(x.shape) < 3:
                std = torch.ones(batch_size, self.reconstructive_dim)
                mean = torch.zeros(batch_size, self.reconstructive_dim)
                # random_reconstructive = torch.normal(mean, std).to("cuda")
                random_reconstructive = mean.to("cuda")
                x = torch.cat((x, random_reconstructive), -1)
            else:
                std = torch.ones(batch_size, self.reconstructive_dim, 1, 1)
                mean = torch.zeros(batch_size, self.reconstructive_dim, 1, 1)
                # random_reconstructive = torch.normal(mean, std).to("cuda")
                random_reconstructive = mean.to("cuda")
                x = torch.cat((x, random_reconstructive), 1)
        x = x.view(x.shape[0], self.latent_dim)
        x = self.d1(x)
        x = x.view(x.shape[0], self.d_max, self.last_size, self.last_size)
        #x = self.deconv1_bn(x)
        x = F.leaky_relu(x, 0.2)

        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)
        x = getattr(self, "deconv%d" % (self.layer_count + 1))(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        if not self.similarity_mode:
            return  mu, logvar, z, self.decode(z.view(-1, self.latent_dim, 1, 1)),
        else:
            return  mu, logvar, z, self.decode(z.view(-1, self.similarity_dim, 1, 1)),

