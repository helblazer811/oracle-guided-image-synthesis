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

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class BasicVAE(nn.Module):

    def __init__(self, z_dim, in_shape, layer_count=4, channels=1, d=128, loss_name="VAETripletLoss", kl_beta=1.0, triplet_beta=1.0, triplet_margin=1.0, num_triplets=None, mask_regularization_beta=0.1, recon_loss_type="mse", recon_beta=1.0):
        super(BasicVAE, self).__init__()
        self.d = d
        self.z_dim = z_dim
        self.in_shape = in_shape
        self.layer_count = layer_count
        self.recon_loss_type = recon_loss_type
        self.channels = channels
        self.loss_name = loss_name
        self.kl_beta = kl_beta
        self.recon_beta = recon_beta
        self.triplet_beta = triplet_beta
        self.num_triplets = num_triplets
        self.triplet_margin = triplet_margin
        self.mask_regularization_beta = mask_regularization_beta
        # run model setup
        self._setup_loss_function()
        self._setup_model()
   
    @classmethod
    def from_config(cls, config):
        return cls(
                z_dim = config["latent_dim"],
                in_shape = config["in_shape"],
                d = config["d"],
                layer_count = config["layer_count"],
                channels = config["channels"],
                loss_name = config["loss_name"],
                recon_loss_type = config["recon_loss_type"] if "recon_loss_type" in config else "mse",
                kl_beta = config["kl_beta"],
                recon_beta = config["recon_beta"] if "recon_beta" in config else 1.0,
                triplet_beta = config["triplet_beta"],
                triplet_margin = config["triplet_margin"],
                mask_regularization_beta = None if not "mask_regularization_beta" in config else config["mask_regularization_beta"],
                num_triplets = None if not "num_triplets" in config else config["num_triplets"],
        )
     
    def _setup_loss_function(self):
        if self.loss_name == "VAETripletLoss":
            self.loss_function = VAETripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta, triplet_margin=self.triplet_margin, recon_loss_type=self.recon_loss_type, recon_beta = self.recon_beta)
        elif self.loss_name == "LogisticTripletLoss":
            self.loss_function = LogisticTripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta)
        elif self.loss_name == "VAEMetricTripletLoss":
            self.loss_function = VAEMetricTripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta, triplet_margin=self.triplet_margin, latent_dim=self.z_dim)
        elif self.loss_name == "UncertaintyTripletLoss":
            self.loss_function = UncertaintyTripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta, latent_dim=self.z_dim)
        elif self.loss_name == "MaskedVAETripletLoss":
            self.loss_function = MaskedVAETripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta, latent_dim=self.z_dim, num_triplets=self.num_triplets)
        elif self.loss_name == "GlobalMaskVAETripletLoss":
            self.loss_function = GlobalMaskVAETripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta, mask_regularization_beta=self.mask_regularization_beta, latent_dim=self.z_dim)
        else:
            raise Exception("Unidentified Loss Function : {}".format(self.loss_name))

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
        self.fc1 = nn.Linear(self.num_linear, self.z_dim)
        self.fc2 = nn.Linear(self.num_linear, self.z_dim)

        self.d1 = nn.Linear(self.z_dim, self.num_linear)

        mul = inputs // self.d // 2

        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, self.d * mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(self.d * mul))
            inputs = self.d * mul
            mul //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, self.channels, 4, 2, 1))
    
   
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def encode(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        for i in range(self.layer_count):
            x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))
        x = x.view(x.shape[0], self.num_linear)
        h1 = self.fc1(x)
        h2 = self.fc2(x)
        return h1, h2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        x = x.view(x.shape[0], self.z_dim)
        x = self.d1(x)
        x = x.view(x.shape[0], self.d_max, self.last_size, self.last_size)
        #x = self.deconv1_bn(x)
        x = F.leaky_relu(x, 0.2)

        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)
        
        x =  getattr(self, "deconv%d" % (self.layer_count + 1))(x)
        x = torch.sigmoid(x)
        # clamp the data
        x = torch.clamp(x, 0, 1)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return  mu, logvar, z, self.decode(z.view(-1, self.z_dim, 1, 1)),

