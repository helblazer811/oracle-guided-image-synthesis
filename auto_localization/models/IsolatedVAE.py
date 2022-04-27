import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import util
from loss.isolated_triplet_loss import IsolatedTripletLoss
from loss.masked_vae_triplet_loss import MaskedVAETripletLoss
from loss.bayesian_triplet_loss import BayesianTripletLoss
from loss.cycle_consistent_triplet_loss import CycleConsistentTripletLoss

class IsolatedVAE(nn.Module):

    def __init__(self, latent_dim, in_shape, similarity_dim=1, reconstructive_dim=1, layer_count=4, channels=1, d=128, kl_beta=1.0, triplet_beta=1.0, triplet_margin=1.0, recon_beta=1.0, loss_name="IsolatedTripletLoss", attributes=[2, 3], masks=False, bce=True, isotropic_variance=False, isolated_warmup=0, cycle_consistent_beta=1.0, similarity_kl_beta=1.0, reparameterize_triplet=False, logistic_triplet=False, pairwise_cycle_consistency=False, logistic_triplet_squared=False, similarity_batchnorm=False, l2_normalization=False, squared_triplet=False, uncertainty_constant=False, sub_loss_type=False):
        super(IsolatedVAE, self).__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.z_dim = self.latent_dim
        self.similarity_dim = similarity_dim
        self.cycle_consistent_beta = cycle_consistent_beta
        self.reconstructive_dim = reconstructive_dim
        if not self.latent_dim == (self.similarity_dim + self.reconstructive_dim):
            raise Exception("Similarity dim and Reconstructive dim do not sum to latent dim")
        self.similarity_mode = False
        self.logistic_triplet = logistic_triplet
        self.similarity_batchnorm = similarity_batchnorm
        self.l2_normalization = l2_normalization
        self.in_shape = in_shape
        self.layer_count = layer_count
        self.channels = channels
        self.sub_loss_type = sub_loss_type
        self.uncertainty_constant = uncertainty_constant
        self.loss_name = loss_name
        self.kl_beta = kl_beta
        self.squared_triplet = squared_triplet
        self.similarity_kl_beta = similarity_kl_beta
        self.attributes = attributes
        self.recon_beta = recon_beta
        self.logistic_triplet_squared = logistic_triplet_squared
        self.pairwise_cycle_consistency = pairwise_cycle_consistency
        self.triplet_beta = triplet_beta
        self.triplet_margin = triplet_margin
        self.reparameterize_triplet = reparameterize_triplet
        self.masks = masks
        self.bce = bce
        self.isotropic_variance = isotropic_variance
        self.isolated_warmup = isolated_warmup
        # run model setup
        self._setup_model()
        self._setup_loss_function()

    def _setup_loss_function(self):
        if self.loss_name == "MaskedVAETripletLoss":
            self.loss_function = MaskedVAETripletLoss(
                kl_beta=self.kl_beta,
                triplet_beta=self.triplet_beta,
                latent_dim=self.z_dim, 
                attributes=self.attributes, 
                similarity_dim=self.similarity_dim,
                component_weighting=self.component_weighting, 
                recon_beta=self.recon_beta
            )
        if self.loss_name == "IsolatedTripletLoss":
            self.loss_function = IsolatedTripletLoss(
                kl_beta=self.kl_beta,
                triplet_beta=self.triplet_beta,
                squared_triplet=self.squared_triplet,
                recon_beta=self.recon_beta, 
                triplet_margin=self.triplet_margin, 
                latent_dim=self.latent_dim, 
                similarity_dim=self.similarity_dim, 
                reconstructive_dim=self.reconstructive_dim, 
                bce=self.bce
             )
        elif self.loss_name == "CycleConsistentTripletLoss":
            self.loss_function = CycleConsistentTripletLoss(
                kl_beta=self.kl_beta,
                similarity_kl_beta=self.similarity_kl_beta,
                triplet_beta=self.triplet_beta,
                recon_beta=self.recon_beta, 
                cycle_consistent_beta=self.cycle_consistent_beta,
                triplet_margin=self.triplet_margin, 
                latent_dim=self.latent_dim, 
                similarity_dim=self.similarity_dim, 
                reconstructive_dim=self.reconstructive_dim, 
                reparameterize_triplet=self.reparameterize_triplet,
                logistic_triplet=self.logistic_triplet,
                logistic_triplet_squared=self.logistic_triplet_squared,
                bce=self.bce
             )
        elif self.loss_name == "BayesianTripletLoss":
            self.loss_function = BayesianTripletLoss(
                kl_beta=self.kl_beta,
                triplet_beta=self.triplet_beta,
                recon_beta=self.recon_beta,
                sub_loss_type=self.sub_loss_type,
                triplet_margin=self.triplet_margin, 
                latent_dim=self.z_dim, 
                similarity_dim=self.similarity_dim,
                reconstructive_dim=self.reconstructive_dim,
                attributes=self.attributes, 
                masks=self.masks, 
                bce=self.bce,
                isolated_warmup=self.isolated_warmup,
                uncertainty_constant=self.uncertainty_constant
             )
        else:
            raise Exception("Unidentified Loss Function : {}".format(self.loss_name))

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
            recon_beta = config["recon_beta"],
            triplet_beta = config["triplet_beta"],
            triplet_margin = config["triplet_margin"],
            sub_loss_type = config["sub_loss_type"] if "sub_loss_type" in config else "bayesian",
            uncertainty_constant = config["uncertainty_constant"] if "uncertainty_constant" in config else False,
            squared_triplet = config["squared_triplet"] if "squared_triplet" in config else False,
            l2_normalization = config["l2_normalization"] if "l2_normalization" in config else False,
            loss_name = config["loss_name"],
            bce = config["bce"] if "bce" in config else True,
            similarity_batchnorm = config["similarity_batchnorm"] if "similarity_batchnorm" in config else False,
            attributes = [2, 3] if not "attributes" in config else config["attributes"],
            masks = False if not "masks" in config else config["masks"],
            isotropic_variance = False if not "isotropic_variance" in config else config["isotropic_variance"],
            isolated_warmup = 0 if not "isolated_warmup" in config else config["isolated_warmup"],
            cycle_consistent_beta = 1.0 if not "cycle_consistent_beta" in config else config["cycle_consistent_beta"],
            reparameterize_triplet = False if not "reparameterize_triplet" in config else config["reparameterize_triplet"],
            logistic_triplet = False if not "logistic_triplet" in config else config["logistic_triplet"],
            pairwise_cycle_consistency = False if not "pairwise_cycle_consistency" in config else config["pairwise_cycle_consistency"],
            logistic_triplet_squared = False if not "logistic_triplet_squared" in config else config["logistic_triplet_squared"],
        )

    def _setup_model(self):
        self.encoder_parameters = []
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
            self.encoder_parameters.append({"params": getattr(self, "conv%d_bn" % (i + 1)).parameters()})
            self.encoder_parameters.append({"params": getattr(self, "conv%d" % (i + 1)).parameters()})

        self.d_max = inputs
        self.last_size = out_sizes[-1][-1]
        self.num_linear = self.last_size ** 2 * self.d_max
        # isolate the linear layers for the recontructive embedding and similarity embedding
        #self.similarity_linear_1 = nn.Linear(self.num_linear, self.num_linear // 2)
        self.similarity_mean_linear = nn.Linear(self.num_linear, self.similarity_dim)
        # Define a batch norm layer
        self.uncertainty_linear = nn.Linear(self.num_linear, 1) # Layer for k constant or variance value
        self.similarity_batchnorm = nn.BatchNorm1d(self.similarity_dim, affine=False)
        self.similarity_logvar_linear = nn.Linear(self.num_linear, self.similarity_dim)
        #self.reconstructive_linear_1 = nn.Linear(self.num_linear, self.num_linear // 2)
        self.reconstructive_mean_linear = nn.Linear(self.num_linear, self.reconstructive_dim)
        self.reconstructive_logvar_linear = nn.Linear(self.num_linear, self.reconstructive_dim)
        #self.encoder_parameters.append({"params": self.similarity_linear_1.parameters()})
        self.encoder_parameters.append({"params": self.similarity_mean_linear.parameters()})
        self.encoder_parameters.append({"params": self.similarity_logvar_linear.parameters()})
        #self.encoder_parameters.append({"params": self.reconstructive_linear_1.parameters()})
        self.encoder_parameters.append({"params": self.reconstructive_mean_linear.parameters()})
        self.encoder_parameters.append({"params": self.reconstructive_logvar_linear.parameters()})

        self.d1 = nn.Linear(self.latent_dim, self.num_linear)

        mul = inputs // self.d // 2

        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, self.d * mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(self.d * mul))
            inputs = self.d * mul
            mul //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, self.channels, 4, 2, 1))

    """
        Returns encoder parameters
    """
    def get_encoder_parameters(self):
        return self.encoder_parameters
    
    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

    def encode(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        for i in range(self.layer_count):
            x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))
        x = x.view(x.shape[0], self.num_linear)
        # use encoding linear layers
        #sim_intermediate = self.similarity_linear_1(x)
        #sim_intermediate = F.relu(sim_intermediate)
        sim_mean = self.similarity_mean_linear(x)
        if self.l2_normalization and self.training:
            sim_mean = F.normalize(sim_mean, dim=-1, p=2)
        # perform batch normalization
        # norm_value = torch.norm(sim_mean, p=2, dim=1).detach()
        # sim_mean = sim_mean.div(norm_value.expand_as(sim_mean))
        if self.similarity_batchnorm:
            sim_mean = self.similarity_batchnorm(sim_mean)
        sim_logvar  = self.similarity_logvar_linear(x)

        if self.similarity_mode:
            return sim_mean, sim_logvar
        #recon_intermediate = self.reconstructive_linear_1(x)
        #recon_intermediate = F.relu(recon_intermediate)
        recon_mean = self.reconstructive_mean_linear(x)
        recon_logvar  = self.reconstructive_logvar_linear(x)
        # concat the outputs 
        mean = torch.cat((sim_mean, recon_mean), -1)
        logvar = torch.cat((sim_logvar, recon_logvar), -1)
        # perform isotropic guassian assignment
        if self.isotropic_variance:
            logvar[:, 1:] = logvar[:, 0].unsqueeze(1)
        # calculate the uncertainty constant
        if self.uncertainty_constant:
            uncertainty_constant = self.uncertainty_linear(x)
            logvar = torch.ones_like(logvar, requires_grad=True) * uncertainty_constant

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

