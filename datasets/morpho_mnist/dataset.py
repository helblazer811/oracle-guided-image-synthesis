import torch
from torch.utils.data import TensorDataset, Dataset
from torchvision import transforms
import json
from PIL import Image
import pandas as pd
import os
import numpy as np
import random
#from util import *
import sys
sys.path.append("../..")
from datasets.morpho_mnist.util import load_idx
from datasets.morpho_mnist.measure import measure_image
from datasets.morpho_mnist.setup_fixed_triplet import save_fixed_triplet_dataset
from auto_localization.oracles.metadata_oracle import MetadataOracle
from auto_localization.oracles.indexed_metadata_oracle import IndexedMetadataOracle

dataset_path = os.path.dirname(__file__)+"/"

def get_indices_for_digits(which_digits, train=True):
    root_dir = dataset_path+"raw-data"
    prefix = "train" if train else "t10k"
    labels_filename = prefix + "-labels-idx1-ubyte.gz"
    lab_inds = load_idx(os.path.join(root_dir, labels_filename)).copy()
    
    indices = []
    for index in range(len(lab_inds)):
        if lab_inds[index] in which_digits:
            indices.append(index)
    return indices

"""
    Dataset that loads the metadata for MorphoMNIST
    dataset.
    - area: Total area/image mass.
    - length: Length of the estimated skeleton.
    - thickness: Mean thickness along the skeleton.
    - slant: Horizontal shear, in radians.
    - width: Width of the bounding parallelogram.
    - height: Height of the bounding parallelogram.
 
"""
class MetadataDataset(TensorDataset):

    def __init__(self, train=True, which_digits=None, apply_transform=False):
        self.train = train
        train_path = dataset_path + "raw-data/train-morpho.csv"
        test_path = dataset_path + "raw-data/t10k-morpho.csv"
        if train:
            dataframe = pd.read_csv(train_path)
            self.column_mins = None
            self.column_maxs = None
        else:
            # this is sketch to ensure same normalization is used for train and test
            temp_metadata = MetadataDataset(train=True, which_digits=which_digits)
            temp_metadata.normalize_data(temp_metadata.metadata)
            self.column_mins = temp_metadata.column_mins
            self.column_maxs = temp_metadata.column_maxs
            dataframe = pd.read_csv(test_path)
        self.apply_transform = apply_transform
        # drop the first column
        dataframe = dataframe.drop(columns=["index"])
        # convert df to numpy
        self.metadata = dataframe.to_numpy()
        # get which_digit_indices 
        self.indices = list(range(0, np.shape(self.metadata)[0]))   
        if not which_digits is None:
            self.indices = get_indices_for_digits(which_digits, train=train)            
        self.metadata = self.metadata[self.indices]
        # normalize
        # self.metadata = self.normalize_data(self.metadata)
    
    """
        Applies transforms to morpho_mnist data
    """
    def transform(self, data):
        # clip data between zero and one 
        clipped = np.clip(data, 0.0, 1.0)
        clipped = clipped + 0.0001
        data = clipped
        # apply sqrt transform to metadata characteristics 0, 1, 2, 4 and inverse sqrt to 5 
        # this was all done to try and remove skewness form data
        if len(np.shape(data)) < 2:
            col_0 = np.sqrt(data[0])
            col_1 = np.sqrt(data[1])
            col_2 = np.sqrt(data[2])
            col_3 = data[3] # identity
            col_4 = np.sqrt(data[4])
            col_5 = 1 - np.sqrt(np.sqrt(1 - data[5]))
            return np.array([col_0, col_1, col_2, col_3, col_4, col_5])
        else: 
            col_0 = np.sqrt(data[:, 0])
            col_1 = np.sqrt(data[:, 1])
            col_2 = np.sqrt(data[:, 2])
            col_3 = data[:, 3] # identity
            col_4 = np.sqrt(data[:, 4])
            col_5 = 1 - np.sqrt(np.sqrt(1 - data[:, 5]))
            # stack the data
            stacked_data = np.stack((col_0, col_1, col_2, col_3, col_4, col_5), axis=-1) 
            return stacked_data

    """
        Utility function to normalize metadata
    """
    def normalize_data(self, data):
        # normalize the data
        if self.column_mins is None or self.column_maxs is None:
            self.column_mins = self.metadata.min(axis=0)
            self.column_maxs = self.metadata.max(axis=0)
        normalized_data = (data - self.column_mins)/(self.column_maxs - self.column_mins)
        if self.apply_transform:
            normalized_data = self.transform(normalized_data)

        return normalized_data

    """
        Utility function to perform morpho_mnist measurement on a digit
    """
    def measure_image(self, image):
        # measure image
        metadata = measure_image(image)
        metadata = list(metadata)
        # normalize metadata
        normalized_metadata = self.normalize_data(metadata)
        return normalized_metadata

    def measure_images(self, images):
        measurements = []
        for image in images:
            metadata = self.measure_image(image)
            measurements.append(metadata)
        measurements = np.array(measurements)
        return measurements

    def __len__(self):
        return np.shape(self.metadata)[0]
        
    def __getitem__(self, idx):
        return self.metadata[idx]

class ImageDataset(TensorDataset):

    def __init__(self, train: bool = True, which_digits=None, grayscale_to_rgb=False):
        self.root_dir = dataset_path+"raw-data"
        #self.transform = transforms.Compose([
        #    transforms.ToPILImage(), 
        #    transforms.Resize((32, 32)),
        #    transforms.ToTensor(),
        #])
        self.grayscale_to_rgb = grayscale_to_rgb
        if not grayscale_to_rgb:
            self.transform = transforms.Compose([
                transforms.Pad(2),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Pad(2),
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            ])

        prefix = "train" if train else "t10k"
        images_filename = prefix + "-images-idx3-ubyte.gz"
        labels_filename = prefix + "-labels-idx1-ubyte.gz"
        im_inds = load_idx(os.path.join(self.root_dir, images_filename)).copy()
        self.images = torch.Tensor(im_inds).float() / 256
        lab_inds = load_idx(os.path.join(self.root_dir, labels_filename)).copy()
        self.labels = torch.from_numpy(lab_inds)
        # get the specified digits  
        self.indices = list(range(0, len(self.labels)))
        if not which_digits is None:
            self.indices = get_indices_for_digits(which_digits, train=train)
        self.images = self.images[self.indices]
        self.labels = self.labels[self.indices]

    def __len__(self):
        return np.shape(self.labels)[0]

    """
        Treats the dataset in a completely unsupervised way
    """
    def __getitem__(self, idx):
        transformed = self.transform(self.images[idx])
        if len(transformed.shape) < 3:
            transformed = transformed.unsqueeze(0)

        return transformed

class TripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """  
     
    def __init__(self, train=True, oracle=None, which_digits=None, one_two_ratio=0.00, dataset_paths=None, single_feature_triplet=False, apply_transform=False, attribute_return=False, mcmv_sampling=False, inject_triplet_noise=0.0, grayscale_to_rgb=False):
        self.train = train
        self.mcmv_sampling = mcmv_sampling
        self.which_digits = which_digits
        self.oracle = oracle
        self.inject_triplet_noise = inject_triplet_noise
        self.grayscale_to_rgb = grayscale_to_rgb
        self.attribute_return = attribute_return
        self.image_dataset = ImageDataset(train=train, which_digits=which_digits, grayscale_to_rgb=grayscale_to_rgb)
        self.metadata_dataset = MetadataDataset(train=train, which_digits=which_digits, apply_transform=apply_transform)
        self.one_two_ratio = one_two_ratio
        self.single_feature_triplet = single_feature_triplet
        self.indexed = False # says that the dataset returns images not indices
        self.transform = None
        self.model = None
        # setup the trian dataset
        self.label_to_indices = {label: np.where(self.image_dataset.labels.numpy() == label)[0]
                                 for label in which_digits}

    """
        Samples a triplet based on the mean cut/max variance spread of the triplet
        in the model latent space
    """
    def mcmv_triplet(self, index, num_samples=20):
        assert not self.model is None
        
        def get_samples():
            samples = []
            for i in range(num_samples):
                anchor_label = self.image_dataset.labels[index].item()
                # get two different random images
                index_a = index
                while index_a == index:
                    index_a = np.random.choice(self.label_to_indices[anchor_label])
                    index_b = index
                while index_b == index or index_b == index_a:
                    index_b = np.random.choice(self.label_to_indices[anchor_label])
                # make a triplet based on their relative morpho_distances
                attribute_index = -1
                if isinstance(self.oracle, IndexedMetadataOracle): 
                    # choose a random attribute index here
                    # get component weighting from the oracle
                    component_weighting = self.oracle.component_weighting
                    nonzero_indices = np.nonzero(component_weighting)[0]
                    attribute_index = np.random.choice(nonzero_indices)
                    choice = self.oracle.answer_query((index, index_a, index_b), single_feature_triplet=self.single_feature_triplet, attribute_index=attribute_index)
                else:
                    anchor = self.metadata_dataset[index]
                    a = self.metadata_dataset[index_a]
                    b = self.metadata_dataset[index_b]
                    choice = self.oracle.answer_query_from_metadata((anchor, a, b), single_feature_triplet=self.single_feature_triplet)
                positive = index_a if choice == 0 else index_b
                negative = index_b if choice == 0 else index_a
                # sample
                samples.append((index, positive, negative, attribute_index))
            
            return samples

        def get_mcmv_scores(samples):
            mcmv_scores = []
            
            for sample in samples:
                # get the embedded values 
                mcmv_score = None 
                # get the images
                index, positive, negative, attribute_index = sample
                positive_image = self.image_dataset[positive].cuda()
                negative_image = self.image_dataset[negative].cuda()
                # calculate the image embedding values from the model
                _, _, positive_z, _ = self.model.forward(positive_image)
                _, _, negative_z, _ = self.model.forward(negative_image)
                mcmv_score = torch.norm(positive_z - negative_z).item()
                # add the score to a list
                mcmv_scores.append(mcmv_score)
            
            mcmv_scores = np.array(mcmv_scores)
            return mcmv_scores
        
        # get samples
        samples = get_samples()
        # get the mcmv scores for the samples
        mcmv_scores = get_mcmv_scores(samples)
        # find the index of the best
        best_index = np.argmax(mcmv_scores) 

        return samples[best_index]

    def one_class_triplet(self, index):
        if self.mcmv_sampling:
            return self.mcmv_triplet(index)
        anchor_label = self.image_dataset.labels[index].item()
        # get two different random images
        index_a = index
        while index_a == index:
            index_a = np.random.choice(self.label_to_indices[anchor_label])
            index_b = index
        while index_b == index or index_b == index_a:
            index_b = np.random.choice(self.label_to_indices[anchor_label])
        # make a triplet based on their relative morpho_distances
        attribute_index = -1
        if isinstance(self.oracle, IndexedMetadataOracle): 
            # choose a random attribute index here
            # get component weighting from the oracle
            component_weighting = self.oracle.component_weighting
            nonzero_indices = np.nonzero(component_weighting)[0]
            attribute_index = np.random.choice(nonzero_indices)
            choice = self.oracle.answer_query((index, index_a, index_b), single_feature_triplet=self.single_feature_triplet, attribute_index=attribute_index, inject_triplet_noise=self.inject_triplet_noise)
        else:
            anchor = self.metadata_dataset[index]
            a = self.metadata_dataset[index_a]
            b = self.metadata_dataset[index_b]
            choice = self.oracle.answer_query_from_metadata((anchor, a, b), single_feature_triplet=self.single_feature_triplet, inject_triplet_noise=self.inject_triplet_noise)
        positive = index_a if choice == 0 else index_b
        negative = index_b if choice == 0 else index_a
        return (index, positive, negative, attribute_index)
    
    def two_class_triplet(self, index):      
        label1 = self.image_dataset.labels[index].item()
        positive_index = index
        attribute_index = -1 
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(set(self.label_to_indices.keys()) - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
        
        return (index, positive_index, negative_index, attribute_index)
    
    def get_image_indices(self, index):
        one_two_random = random.uniform(0, 1) 
        
        if one_two_random > self.one_two_ratio:
            # do one class triplet
            anchor_index, positive_index, negative_index, attribute_index = self.one_class_triplet(index)
        else:
            # do two class triplet
            anchor_index, positive_index, negative_index, attribute_index = self.two_class_triplet(index)
        
        return (anchor_index, positive_index, negative_index, attribute_index)
   
    def __getitem__(self, index):
        index = index % len(self.image_dataset)
        one_two_random = random.uniform(0, 1) 
        anchor_index, positive_index, negative_index, attribute_index = self.get_image_indices(index) 
        anchor = self.image_dataset[anchor_index]
        positive = self.image_dataset[positive_index]
        negative = self.image_dataset[negative_index]

        if self.attribute_return:
            return (anchor, positive, negative, attribute_index), []
        else:
            return (anchor, positive, negative), []

    def __len__(self):
        return len(self.image_dataset)

class FixedTripletDataset(Dataset):
    """
        Same as TripletDataset except the triplets are pre-generated and loaded through a 
        given csv file
    """  
     
    def __init__(self, train=True, oracle=None, which_digits=None, one_two_ratio=0.00, triplets_per_image=20, load_from_file=True, apply_transform=False, data_path=None, attributes=False):
        self.one_two_ratio = one_two_ratio
        self.which_digits = which_digits
        self.train = train
        self.indexed = True # describes whether or not the dataset returns indices or images
        self.transform = None
        self.image_dataset = ImageDataset(train=train, which_digits=which_digits)
        self.metadata_dataset = MetadataDataset(train=train, which_digits=which_digits, apply_transform=apply_transform)
        self.triplets_per_image = triplets_per_image
        self.attributes = attributes
        self.oracle = oracle

        if data_path is None:
            if train:
                self.dataset_path = dataset_path + "derived_data/fixed_triplet_train.csv"
            else:
                self.dataset_path = dataset_path + "derived_data/fixed_triplet_test.csv"
        else:
            self.dataset_path = data_path

        if not load_from_file:
            # load a regular triplet dataset
            self.base_triplet_dataset = TripletDataset(train=self.train, oracle=self.oracle, which_digits=self.which_digits, one_two_ratio=one_two_ratio) 
            # setup a fixed triplet dataset
            save_fixed_triplet_dataset(self.base_triplet_dataset, self.dataset_path, triplets_per_example=self.triplets_per_image)
            # load up the dataframe

        self.dataframe = self._load_dataframe(self.dataset_path)
    
    def _load_dataframe(self, data_path):
        # load the dataframe
        df = pd.read_csv(data_path)
        # return the dataframe
        return df
 
    def __getitem__(self, index):
        # load the index for the dataframe 
        triplet_row = self.dataframe.iloc[index]
        anchor_index = triplet_row["anchor"]
        positive_index = triplet_row["positive"]
        negative_index = triplet_row["negative"]
        # convert the index to an image
        anchor = self.image_dataset[anchor_index]
        positive = self.image_dataset[positive_index]
        negative = self.image_dataset[negative_index]
        if self.attributes:
            attribute_index = triplet_row["attribute_index"]
            return (anchor, positive, negative, attribute_index), []
        else:
            return (anchor, positive, negative), []

    def __len__(self):
        return len(self.dataframe.index)
