from __future__ import division

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from loss.vae_loss import VAELoss
from loss.masked_vae_triplet_loss import MaskedVAETripletLoss
from loss.isolated_triplet_loss import IsolatedTripletLoss

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.05)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.05)
        torch.nn.init.constant_(m.bias.data, 0.0)

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class CelebABetaVAE(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3, img_size=32, kl_beta=1.0, triplet_beta=1.0, loss_name="", attributes=[2, 3], similarity_dim=2, reconstructive_dim=2, bce=False):
        super(CelebABetaVAE, self).__init__()
        self.z_dim = z_dim
        self.kl_beta = kl_beta
        self.triplet_beta = triplet_beta
        self.attributes = attributes
        self.bce = bce
        self.similarity_dim = similarity_dim
        self.reconstructive_dim = reconstructive_dim
        self.img_size = img_size
        self.num_filters = 32
        self.nc = nc
        self.loss_name = loss_name
        #self.encoder = ConvEncoder(self.z_dim * 2, self.nc, self.img_size, False, num_filters=64)
        #self.decoder = ConvDecoder(self.z_dim, self.nc, self.img_size, num_filters=64)
        if img_size == 64: 
            self.encoder = nn.Sequential(
                nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
                nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
                nn.ReLU(True),
                nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
                nn.ReLU(True),
                View((-1, 256*1*1)),                 # B, 256
                nn.Linear(256, z_dim*2),             # B, z_dim*2
            )
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, 256),               # B, 256
                View((-1, 256, 1, 1)),               # B, 256,  1,  1
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
                nn.ReLU(True),
                nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(int(nc), self.num_filters, 4, 2, 1),
                nn.ReLU(True),
                nn.Conv2d(self.num_filters, self.num_filters * 2, 4, 2, 1),
                nn.ReLU(True),
                nn.Conv2d(self.num_filters * 2, self.num_filters * 2, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(self.num_filters * 2, self.num_filters * 2, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(self.num_filters * 2, self.num_filters * 2, 4, 2, 1),
                nn.ReLU(True),
                nn.Conv2d(self.num_filters * 2, self.num_filters, 4, 2, 1),
                View((-1, self.num_filters * (img_size // (2 ** 4)) * (img_size // (2 ** 4)))),
                nn.Linear(self.num_filters * (img_size // (2 ** 4)) * (img_size // (2 ** 4)), z_dim * 2)
            )

            self.decoder = nn.Sequential(
                nn.Linear(z_dim, self.num_filters * (img_size // (2 ** 4)) * (img_size // (2 ** 4))),
                nn.ReLU(True),
                View((-1, self.num_filters * (img_size // (2 ** 4)) * (img_size // (2 ** 4)), 1, 1)),
                nn.ConvTranspose2d(self.num_filters, self.num_filters * 2, 3),
                nn.ReLU(True),
#                nn.ConvTranspose2d(self.num_filters * 2, self.num_filters * 2, 3),
#                nn.ReLU(True),
#                nn.ConvTranspose2d(self.num_filters * 2, self.num_filters * 2, 3, padding=1),
#                nn.ReLU(True),
#                nn.ConvTranspose2d(self.num_filters * 2, self.num_filters, 3, padding=1),
#                nn.ReLU(True),
#                nn.ConvTranspose2d(self.num_filters, self.num_filters, 3),
#                nn.ReLU(True),
#                nn.ConvTranspose2d(self.num_filters, int(nc), 4, stride=2),
            )
            """
           self.encoder = nn.Sequential(
                nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 14, 14
                nn.ReLU(True),
                nn.Conv2d(32, 64, 3, 2, 1),          # B,  32, 7, 7
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 1),             # B,  64,  4,  4
                nn.ReLU(True),
                nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
                nn.ReLU(True),
                View((-1, 256*1*1)),                 # B, 256
                nn.Linear(256, z_dim*2),             # B, z_dim*2
            )
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, 256),               # B, 256
                View((-1, 256, 1, 1)),               # B, 256,  1,  1
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 64, 4, 1, 0),   # B,  64,  4,  4
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 1, 0), # B,  32,  7,  7
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, 4, 2, 0), # B,  32, 14, 14
                nn.ReLU(True),
                nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 28, 28
            )            
        """
        #self.weight_init()
        # setup the loss function
        self._setup_loss_function()

    @classmethod
    def from_config(cls, config):
        return cls(
            z_dim = config["latent_dim"],
            kl_beta = config["kl_beta"],
            triplet_beta = config["triplet_beta"],
            img_size = config["img_size"],
            loss_name = config["loss_name"], 
            reconstructive_dim = config["reconstructive_dim"] if "reconstructive_dim" in config else 2,
            similarity_dim = config["similarity_dim"] if "similarity_dim" in config else 2,
            attributes = config["attributes"] if "attributes" in config else [0, 1],
            bce = config["bce"] if "bce" in config else True,
        )

    def _setup_loss_function(self):
        if self.loss_name == "VAELoss":
            self.loss_function = VAELoss(kl_beta=self.kl_beta)
        elif self.loss_name == "MaskedVAETripletLoss":
            self.loss_function = MaskedVAETripletLoss(triplet_beta=self.triplet_beta, kl_beta=self.kl_beta, attributes=self.attributes, latent_dim=self.z_dim, bce=self.bce)
        elif self.loss_name == "IsolatedTripletLoss":
            self.loss_function = IsolatedTripletLoss(kl_beta=self.kl_beta, triplet_beta=self.triplet_beta, similarity_dim=self.similarity_dim, reconstructive_dim=self.reconstructive_dim)
        else:
            raise Exception("Unidentified Loss Function : {}".format(self.loss_name))

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        print(x.shape)
        distributions = self._encode(x)
        print(distributions.shape)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparameterize(mu, logvar)
        x_recon = self._decode(z)

        return  mu, logvar, z, x_recon

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def encode(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        
        return mu, logvar

    def decode(self, z):
        return self._decode(z)

class ConvEncoder(nn.Module):

    def __init__(self, z_dim, c_dim, img_size, norm_ae_flag, num_filters=64):
        """
        Encoder initializer
        :param x_dim: dimension of the input
        :param z_dim: dimension of the latent representation
        :param M: number of transport operators
        """
        super(ConvEncoder, self).__init__()
        self.num_filters = num_filters
        self.img_size = img_size
        if self.img_size == 32:
            self.main = nn.Sequential(
                nn.Conv2d(int(c_dim), self.num_filters, 4, 2, 1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(self.num_filters, self.num_filters * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.num_filters * 2),
                nn.ReLU(True),
                nn.Conv2d(self.num_filters * 2, self.num_filters * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.num_filters * 2),
                nn.ReLU(True),
                nn.Conv2d(self.num_filters * 2, self.num_filters * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.num_filters * 2),
                nn.ReLU(True),
                nn.Conv2d(self.num_filters * 2, self.num_filters * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.num_filters * 2),
                nn.ReLU(True),
                nn.Conv2d(self.num_filters * 2, self.num_filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.num_filters),
                nn.ReLU(True)
            )
            self.fc = nn.Linear(self.num_filters * (img_size // (2 ** 4)) * (img_size // (2 ** 4)), z_dim)
        elif self.img_size == 64:
            self.main = nn.Sequential(
                nn.Conv2d(int(c_dim), self.num_filters // 4, 4, 2, 1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(self.num_filters // 4, self.num_filters // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.num_filters // 2),
                nn.ReLU(True),
                nn.Conv2d(self.num_filters // 2, self.num_filters, 3, 2, 1, bias=False),
                nn.BatchNorm2d(self.num_filters),
                nn.ReLU(True),
                nn.Conv2d(self.num_filters, self.num_filters * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.num_filters * 2),
                nn.ReLU(True),
                nn.Conv2d(self.num_filters * 2, self.num_filters * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.num_filters * 2),
                nn.ReLU(True),
                nn.Conv2d(self.num_filters * 2, self.num_filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.num_filters),
                nn.ReLU(True)
            )
            self.fc = nn.Linear(self.num_filters * (img_size // (2 ** 5)) * (img_size // (2 ** 5)), z_dim)
        else:
            self.model_enc = nn.Sequential(
                nn.Conv2d(int(c_dim), num_filters, 4, stride=2, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, 4, stride=2, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(num_filters, num_filters, 4, stride=1, padding=0),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
            )

            self.fc_mean = nn.Linear(int(num_filters*img_size*img_size/16),z_dim)
        self.norm_ae_flag = norm_ae_flag

    def forward(self, x):
         # 2 hidden layers encoder
        if self.img_size == 32 or self.img_size == 64:
            x = self.main(x)
            x = x.view(x.size(0),-1)
            z_mean = self.fc(x)
        else:
            x = self.model_enc(x)
            x = x.view(x.size(0),-1)
            z_mean = self.fc_mean(x)
        if self.norm_ae_flag == 1:
            z_mean = F.normalize(z_mean)
        return z_mean

class ConvDecoder(nn.Module):

    def __init__(self, z_dim, c_dim, img_size, num_filters=64):
        super(ConvDecoder, self).__init__()
        self.num_filters = num_filters
        self.img_size = img_size
        if self.img_size == 28:
            self.img_4 = img_size/4
        elif self.img_size == 32:
            self.img_4 = 9
        elif self.img_size == 64:
            self.img_4 = 25

        if self.img_size == 32 or self.img_size == 64:
            self.proj = nn.Sequential(
                nn.Linear(z_dim, self.num_filters * self.img_4 * self.img_4),
                nn.ReLU()
            )
            self.main = nn.Sequential(
                # 9x9
                # H/W + 2
                nn.ConvTranspose2d(self.num_filters, self.num_filters * 2, 3),
                nn.BatchNorm2d(self.num_filters * 2),
                nn.ReLU(True),
                # H/W + 2
                nn.ConvTranspose2d(self.num_filters * 2, self.num_filters * 2, 3),
                nn.BatchNorm2d(self.num_filters * 2),
                nn.ReLU(True),
                # H/W + 0
                nn.ConvTranspose2d(self.num_filters * 2, self.num_filters * 2, 3, padding=1),
                nn.BatchNorm2d(self.num_filters * 2),
                nn.ReLU(True),
                # H/W + 0
                nn.ConvTranspose2d(self.num_filters * 2, self.num_filters, 3, padding=1),
                nn.BatchNorm2d(self.num_filters),
                nn.ReLU(True),
                # H/W + 2
                nn.ConvTranspose2d(self.num_filters, self.num_filters, 3),
                nn.BatchNorm2d(self.num_filters),
                nn.ReLU(True),
                # 15x15
                # H/W*2 + 2
                nn.ConvTranspose2d(self.num_filters, int(c_dim), 4, stride=2),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                    nn.Linear(z_dim,int(self.img_4*self.img_4*num_filters)),
                    nn.ReLU(),
                    )

            self.model = nn.Sequential(
                nn.ConvTranspose2d(num_filters, num_filters, 4, stride=1, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.ConvTranspose2d(num_filters, num_filters, 4, stride=2, padding=2),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.ConvTranspose2d(num_filters, int(c_dim), 4, stride=2, padding=1),
                nn.Sigmoid()
            )

    def forward(self, z):
        batch_size = z.size()[0]
        if self.img_size == 32 or self.img_size == 64:
            temp_var = self.proj(z)
            temp_var = temp_var.view(batch_size, self.num_filters, int(self.img_4), int(self.img_4))
            img = self.main(temp_var)
        else:
            temp_var = self.fc(z)
            temp_var = temp_var.view(batch_size, self.num_filters, int(self.img_4), int(self.img_4))
            img = self.model(temp_var)
        return img

