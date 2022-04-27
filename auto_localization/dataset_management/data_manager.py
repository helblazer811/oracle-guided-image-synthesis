import torch
from torch.utils.data import BatchSampler, SequentialSampler
import torchvision
import os
import random
import numpy as np
import sys
sys.path.append("..")

import pandas as pd
from PIL import Image

source_dir = "/home/alec/latent-space-localization/source/morpho_mnist"

"""
    Generic data manager object for
    handling train/test split, data loading, 
    holding triplets, and holding metadata. 
    This object can take in any datasets
"""
class DataManager():

    def __init__(self, image_datasets, triplet_datasets, metadata_datasets=None, batch_size=64, triplet_batch_size=None, num_workers=0):
        self.image_train, self.image_test = image_datasets
        self.triplet_train, self.triplet_test = triplet_datasets
        if not metadata_datasets is None:
            self.metadata_train, self.metadata_test = metadata_datasets
        self.batch_size = batch_size
        if triplet_batch_size is None:
            self.triplet_batch_size = self.batch_size
        else:
            self.triplet_batch_size = triplet_batch_size
        self.num_workers = num_workers
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_train_loader = None
        self.image_test_loader = None
        self.triplet_train_loader = None
        self.triplet_test_loader = None
        # setup code
        self.setup_data_loaders()
    
    def setup_data_loaders(self, triplet_mining=False):
        kwargs = {'pin_memory': True} if self.device == "cuda" else {}
        # make triplet loaders
        self.triplet_train_loader = torch.utils.data.DataLoader(self.triplet_train, batch_size=self.triplet_batch_size, num_workers=self.num_workers, shuffle=False, **kwargs)
        self.triplet_test_loader = torch.utils.data.DataLoader(self.triplet_test, batch_size=self.triplet_batch_size, num_workers=self.num_workers, shuffle=False, **kwargs)
        # make train loaders
        self.image_train_loader = torch.utils.data.DataLoader(self.image_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, **kwargs)
        self.image_test_loader = torch.utils.data.DataLoader(self.image_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, **kwargs)

    def get_latent_point_training(self, gen_model, inputs=None, test=True, num_max=2000, variances=False):
        # set default inputs
        if test:
            images = self.image_test
        else:
            images = self.image_train
        
        with torch.no_grad():
            if inputs is None:
                z_vals = []
                logvars = []
                max_index = min(num_max, len(images))
                for i in range(max_index):
                    z_mean, z_logvar = gen_model.encode(images[i].to(self.device))
                    z_vals.append(z_mean)
                    logvars.append(z_logvar)
                z_train = torch.cat(z_vals).detach().cpu().numpy()
                logvars = torch.cat(logvars).detach().cpu().numpy()

                if variances:
                    return z_train, logvars

                return z_train
            
            inputs = inputs.unsqueeze(1)
            z_mean, z_logvar = gen_model.encode(inputs.to(self.device))
        # z_train = gen_model.sample(z_mean.to(self.device), z_logvar.to(self.device)).detach().cpu().numpy()
        if variances:
            return z_mean, z_logvar

        return z_mean
    
