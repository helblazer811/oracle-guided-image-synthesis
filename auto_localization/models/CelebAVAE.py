import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import util
from loss.vae_triplet_loss import VAETripletLoss
from loss.metric_triplet_loss import VAEMetricTripletLoss
from loss.uncertainty_triplet_loss import UncertaintyTripletLoss
from loss.masked_vae_triplet_loss import MaskedVAETripletLoss
from loss.global_mask_vae_triplet_loss import GlobalMaskVAETripletLoss
from loss.logistic_triplet_loss import LogisticTripletLoss
from loss.isolated_triplet_loss import IsolatedTripletLoss
from loss.bayesian_triplet_loss import BayesianTripletLoss

class CelebAVAE(nn.Module):

    def __init__(self, z_dim=32, loss_name="VAETripletLoss", kl_beta=1.0, triplet_beta=1.0, triplet_margin=1.0, num_triplets=None, in_shape=64, similarity_dim=2, reconstructive_dim=2, similarity_mode=False, latent_dim=None, reparam_lambda=1.0, recon_beta=1.0, masks=False, attributes=[-1], bce=True, channels=3, init_zero_logvar_layer=False, sub_loss_type="isolated"):
        super(CelebAVAE, self).__init__()
        self.z_dim = z_dim
        self.latent_dim = self.z_dim
        self.in_shape = in_shape
        self.loss_name = loss_name
        self.kl_beta = kl_beta
        self.recon_beta = recon_beta
        self.reparam_lambda = reparam_lambda
        self.sub_loss_type = sub_loss_type
        self.triplet_beta = triplet_beta
        self.similarity_dim = similarity_dim
        self.similarity_mode = similarity_mode
        self.reconstructive_dim = reconstructive_dim
        self.channels = channels
        self.masks = masks
        self.init_zero_logvar_layer = init_zero_logvar_layer
        self.bce = bce
        self.num_triplets = num_triplets
        self.triplet_margin = triplet_margin
        self.attributes = attributes
        # run model setup
        self._setup_loss_function()
        self._setup_model()
   
    @classmethod
    def from_config(cls, config):
        return cls(
            z_dim = config["latent_dim"],
            in_shape = config["in_shape"],
            loss_name = config["loss_name"],
            kl_beta = config["kl_beta"],
            triplet_beta = config["triplet_beta"],
            recon_beta = config["recon_beta"], 
            triplet_margin = config["triplet_margin"],
            similarity_dim = config["similarity_dim"],
            channels = config["channels"] if "channels" in config else 3,
            masks = config["masks"], 
            sub_loss_type = config["sub_loss_type"],
            reconstructive_dim = config["reconstructive_dim"],
            num_triplets = None if not "num_triplets" in config else config["num_triplets"],
            attributes = [-1] if not "attributes" in config else config["attributes"],
            bce = True if not "bce" in config else config["bce"],
            init_zero_logvar_layer = True if "init_zero_logvar_layer" in config else config["init_zero_logvar_layer"],
        )
     
    def _setup_loss_function(self):
        if self.loss_name == "VAETripletLoss":
            self.loss_function = VAETripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta, triplet_margin=self.triplet_margin)
        elif self.loss_name == "LogisticTripletLoss":
            self.loss_function = LogisticTripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta)
        elif self.loss_name == "IsolatedTripletLoss":
            self.loss_function = IsolatedTripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta, similarity_dim=self.similarity_dim, reconstructive_dim=self.reconstructive_dim, triplet_margin=self.triplet_margin, recon_beta=self.recon_beta)
        elif self.loss_name == "VAEMetricTripletLoss":
            self.loss_function = VAEMetricTripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta, triplet_margin=self.triplet_margin, latent_dim=self.z_dim)
        elif self.loss_name == "UncertaintyTripletLoss":
            self.loss_function = UncertaintyTripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta, latent_dim=self.z_dim)
        elif self.loss_name == "MaskedVAETripletLoss":
            self.loss_function = MaskedVAETripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta, latent_dim=self.z_dim, num_triplets=self.num_triplets)
        elif self.loss_name == "GlobalMaskVAETripletLoss":
            self.loss_function = GlobalMaskVAETripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta, mask_regularization_beta=self.mask_regularization_beta, latent_dim=self.z_dim)
        elif self.loss_name == "BayesianTripletLoss":
            self.loss_function = BayesianTripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta, latent_dim=self.z_dim, similarity_dim=self.similarity_dim, attributes=self.attributes, triplet_margin=self.triplet_margin, masks=self.masks, recon_beta=self.recon_beta, bce=self.bce, sub_loss_type=self.sub_loss_type)
        else:
            raise Exception("Unidentified Loss Function : {}".format(self.loss_name))

    def _setup_model(self):
        self.encoder_parameters = []
        modules = []
        if self.in_shape == (128, 128) or self.in_shape == 128:
            self.hidden_dims = [32, 64, 128, 256, 512, 1028]
        elif self.in_shape == (64, 64) or self.in_shape == 64:
            self.hidden_dims = [32, 64, 128, 256, 512]
        elif self.in_shape == (32, 32) or self.in_shape == 32:
            self.hidden_dims = [32, 64, 128, 256]
        else:
            raise Exception(f"Unsupported input shape for CelebAVAE : {self.in_shape}")
        #self.hidden_dims = [32, 64, 128, 256]
        #mult = 4 if self.in_shape == 64 else 1 
        in_channels = self.channels
        mult = 4
        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
            self.encoder_parameters.append({"params": modules[-1].parameters()})

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1]*mult, self.z_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1]*mult, self.z_dim)

        self.encoder_parameters.append({"params": self.fc_mu.parameters()})
        self.encoder_parameters.append({"params": self.fc_var.parameters()})
        if self.init_zero_logvar_layer:
            nn.init.zeros_(self.fc_var.weight)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.z_dim, self.hidden_dims[-1]*mult)
        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[-1],
                               self.hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dims[-1], out_channels=self.channels,
                      kernel_size= 3, padding= 1),
            nn.Tanh()
        )

        self.hidden_dims.reverse()

    """
        Returns encoder parameters
    """
    def get_encoder_parameters(self):
        return self.encoder_parameters
   
    def encode(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        if self.similarity_mode:
            return mu[:, 0:self.similarity_dim], log_var[:, 0:self.similarity_dim]

        return mu, log_var

    def reparameterize(self, mu, logvar):
        if self.training:
            var = torch.exp(logvar)
            # perform reparameterization
            dim_reparams = torch.ones(self.latent_dim).cuda()
            dim_reparams[self.similarity_dim:] = self.reparam_lambda
            mu = dim_reparams * mu
            var = dim_reparams * var + (1 - dim_reparams)
            std = var ** 0.5
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, z.shape[-1])
        if self.similarity_mode:
            new_z = torch.zeros(z.shape[0], self.latent_dim).cuda()
            new_z[:, 0:self.similarity_dim] = z
            z = new_z
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        if self.bce:
            result = torch.clamp(result, 0, 1)
        return result

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return  mu, logvar, z, self.decode(z)

