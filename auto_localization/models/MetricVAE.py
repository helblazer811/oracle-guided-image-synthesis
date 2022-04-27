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

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class MetricVAE(nn.Module):

    def __init__(self, z_dim, in_shape, layer_count=4, channels=1, d=128, loss_name="VAETripletLoss", kl_beta=1.0, triplet_beta=1.0, triplet_margin=1.0):
        super(MetricVAE, self).__init__()
        self.d = d
        self.z_dim = z_dim
        self.in_shape = in_shape
        self.layer_count = layer_count
        self.channels = channels
        self.loss_name = loss_name
        self.kl_beta = kl_beta
        self.triplet_beta = triplet_beta
        self.triplet_margin = triplet_margin
        self.use_metric = False
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
                kl_beta = config["kl_beta"],
                triplet_beta = config["triplet_beta"],
                triplet_margin = config["triplet_margin"],
        )
     
    def _setup_loss_function(self):
        self.loss_function = VAEMetricTripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta, triplet_margin=self.triplet_margin, latent_dim=self.z_dim)

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
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        for i in range(self.layer_count):
            x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))
        x = x.view(x.shape[0], self.num_linear)
        mean = self.fc1(x)
        logvar = self.fc2(x)
        # if use_metric is toggled then do a mapping with it
        if self.use_metric:
            similarity_embedding = self.loss_function.metric_linear(mean)
            return similarity_embedding, similarity_embedding

        return mean, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        if self.use_metric:
            print("using metric")
            self.loss_function.setup_inverse()
            x = x.view(x.shape[0], self.z_dim)
            x = self.loss_function.inverse_metric_linear(x)
        x = x.view(x.shape[0], self.z_dim)
        x = self.d1(x)
        x = x.view(x.shape[0], self.d_max, self.last_size, self.last_size)
        #x = self.deconv1_bn(x)
        x = F.leaky_relu(x, 0.2)

        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)

        x = torch.sigmoid(getattr(self, "deconv%d" % (self.layer_count + 1))(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return  mu, logvar, z, self.decode(z.view(-1, self.z_dim, 1, 1)),

