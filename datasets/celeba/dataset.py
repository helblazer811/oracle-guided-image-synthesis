import torch
from torch.utils.data import TensorDataset, Dataset
from torchvision import transforms, datasets
import json
from PIL import Image
import pandas as pd
import os
import numpy as np
import random
#from util import *
import sys
sys.path.append(os.environ["LATENT_PATH"])

class TripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """  
     
    def __init__(self, train=True, oracle=None, which_attributes=None, input_shape=(64, 64), dataset_paths=None, single_feature_triplet=True):
        self.train = train
        self.which_attributes = which_attributes 
        self.single_feature_triplet = single_feature_triplet
        self.input_shape = input_shape
        self.oracle = oracle
        self.metadata_dataset = None
        self.image_dataset = ImageDataset(train=train, input_shape=self.input_shape)
        self.transform = None
        self.indexed = False
            
    def generate_triplet(self, index, single_feature_triplet=False, attribute_index=-1):
        if callable(getattr(self.oracle, "generate_triplet", None)):
            return self.oracle.generate_triplet(single_feature_triplet=single_feature_triplet, attribute_index=attribute_index)
        # get two different random images
        index_a = np.array(index).astype(np.int_)
        while index_a == index:
            index_a = np.random.choice(np.arange(0, len(self.image_dataset))).astype(np.int_)
            index_b = index
        while index_b == index or index_b == index_a:
            index_b = np.random.choice(np.arange(0, len(self.image_dataset))).astype(np.int_)
        # attribute index randomness
        if len(np.shape(index)) == 0: 
            N = 1 
        else:
            N = len(index)
       
        if self.which_attributes is None:
            attribute_index = -1
        else:
            attribute_index = np.random.choice(self.which_attributes, size=N)
        # make a triplet based on their relative morpho_distances
        if self.oracle.indexed:
            choice = self.oracle.answer_query((index, index_a, index_b), 
                                        single_feature_triplet=self.single_feature_triplet, attribute_index=attribute_index)
        else:
            anchor = self.image_dataset[index]
            a = self.image_dataset[index_a]
            b = self.image_dataset[index_b]
            choice = self.oracle.answer_query((anchor, a, b), 
                                        single_feature_triplet=self.single_feature_triplet, attribute_index=attribute_index)
        positive = index_a if choice == 0 else index_b
        negative = index_b if choice == 0 else index_a

        return (index, positive, negative, attribute_index)

    def __len__(self):
        return len(self.image_dataset)
     
    def __getitem__(self, index):
        anchor_index, positive_index, negative_index, attribute_index = self.generate_triplet(index)
        anchor = self.image_dataset[anchor_index]
        positive = self.image_dataset[positive_index]
        negative = self.image_dataset[negative_index]
        
        return (anchor, positive, negative, attribute_index), []
   
class CelebAMetadataDataset(TensorDataset):

    def __init__(self, train: bool = True, which_attributes=[]):
        self.root_dir = os.environ.get("LATENT_PATH")+"/datasets/celeba/raw_data"
        self.which_attributes = which_attributes
        train_string = 'train' if train else 'test'
        self.transform = transforms.Compose([])
        self.dataset = datasets.CelebA(self.root_dir, train_string, download=False)

    def __len__(self):
        return len(self.dataset)

    """
        Treats the dataset in a completely unsupervised way
    """
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        if len(target.shape) < 2:
            target = target[None, :] 
        target = target[:, self.which_attributes]
        return target


class ImageDataset(TensorDataset):

    def __init__(self, input_shape=(64, 64), train: bool = True, indexed = False):
        self.root_dir = os.environ.get("LATENT_PATH")+"/datasets/celeba/raw_data"
        self.input_shape = input_shape
        self.indexed = indexed
        train_string = 'train' if train else 'test'
#        SetRange = transforms.Lambda(lambda img: 2 * img - 1.)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(148),
            transforms.Resize(input_shape),
            transforms.ToTensor(),
#            SetRange(),
        ])
        self.dataset = datasets.CelebA(self.root_dir, train_string, download=False)

    def __len__(self):
        return len(self.dataset)

    """
        Treats the dataset in a completely unsupervised way
    """
    def __getitem__(self, idx):
        if isinstance(idx, list) or isinstance(idx, np.ndarray):
            images = []
            for index in idx:
                image, target = self.dataset[index]
                transformed = self.transform(image)
                images.append(transformed)
            images = torch.stack(images)
            if self.indexed:
                return images, idx
            else:
                return images

        image, target = self.dataset[idx]
        transformed = self.transform(image)
        if self.indexed:
            return transformed, idx
        else:
            return transformed

