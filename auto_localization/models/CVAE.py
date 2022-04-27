import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def _weights_init(m):
    classname = m.__class__.__name__
    if 'Conv2d' in classname:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)

def sample_noise(num, dim, device=None) -> torch.Tensor:
    return torch.randn(num, dim, device=device)

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, out_size: int):
        super().__init__()
        self.out_size = out_size
        self.latent_dim = latent_dim
        self.h1_dim = 1024
        self.fc1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.h1_dim),
            nn.BatchNorm1d(self.h1_dim),
            nn.ReLU(inplace=True)
        )
        self.h2_nchan = 128
        h2_dim = 7 * 7 * self.h2_nchan
        self.fc2 = nn.Sequential(
            nn.Linear(self.h1_dim, h2_dim),
            nn.BatchNorm1d(h2_dim),
            nn.ReLU(inplace=True)
        )
        self.h3_nchan = 64
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(self.h2_nchan, self.h3_nchan,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.h3_nchan),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(self.h3_nchan, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x).view(-1, self.h2_nchan, 7, 7)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, latent_dim: int, out_size: int):
        super().__init__()
        self.out_size = out_size
        self.h1_nchan = 64
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, self.h1_nchan, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(.1, inplace=True)
        )
        self.h2_nchan = 128
        self.conv2 = nn.Sequential(
                nn.Conv2d(self.h1_nchan, self.h2_nchan, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(self.h2_nchan),
                nn.LeakyReLU(.1, inplace=True)
        )
        self.h3_dim = 1024
        self.fc1 = nn.Sequential(
                nn.Linear(7 * 7 * self.h2_nchan, self.h3_dim),
                nn.BatchNorm1d(self.h3_dim),
                nn.LeakyReLU(.1, inplace=True)
        )
        self.fc2_mean = nn.Linear(self.h3_dim, latent_dim)
        self.fc2_logvar = nn.Linear(self.h3_dim, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x).view(-1, 7 * 7 * self.h2_nchan)
        x = self.fc1(x)
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar

class VAE(nn.Module):
    def __init__(self, latent_dim: int, out_size: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_size = out_size

        self.enc = Encoder(latent_dim, self.out_size)
        self.dec = Decoder(latent_dim, self.out_size)

        self.apply(_weights_init)

    def sample_latent(self, num: int):
        return sample_noise(num, self.latent_dim, self.device)

    def sample_posterior(self, data, num: int = 1):
        noise = torch.randn(data.shape[0], num, self.latent_dim, device=self.device)
        mean, logvar = self.enc(data)
        latent = mean.unsqueeze(1) + (.5 * logvar).exp().unsqueeze(1) * noise

    def forward(self, data):
        noise = self.sample_latent(data.shape[0])
        mean, logvar = self.enc(data)
        latent = mean + (.5 * logvar).exp() * noise
        recon = self.dec(latent)
        return mean, logvar, latent, recon

    #interface function
    def encode(self, x):
        x = x.float()
        mean, log_var = self.enc(x.view(-1, 1, self.out_size, self.out_size))
        return mean, log_var

    #interface function
    def sample(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps.mul(std)

    #interface function
    def decode(self, z):
        if not type(z).__name__ == "Tensor":
            if np.shape(z)[0] < 2:
                tensor_input = torch.tensor([z, z])
                tensor_input = tensor_input.type(torch.FloatTensor).squeeze()
                decoded = self.dec(tensor_input)
                return decoded[0]
            else:
                tensor_input = torch.tensor(z)
                tensor_input = tensor_input.type(torch.FloatTensor)
                decoded = self.dec(tensor_input)
                return decoded
        decoded = self.dec(z)
        return decoded

    @property
    def device(self):
        return next(self.parameters()).device

