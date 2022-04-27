import torch
from torch import nn
import numpy as np
import sys
import os
sys.path.append(os.environ["LATENT_PATH"])
from datasets.morpho_mnist.measure import measure_image, measure_batch

def normalize_data(data):
    data = data.numpy()
    means = np.mean(data, axis=0)
    mins = np.amin(data, 0)
    maxs = np.amax(data, 0)

    return torch.Tensor((data - means)/np.abs(maxs - mins))

"""
    This class acts as a dummy gen_model to be passed to the localizer
    and then subsequently MorphoDataManager. It implements the sample and encode
    methods in a hardcoded way by treating the image metadata vectors as 
"""
class DummyModel(nn.Module):

    def __init__(self):
        super(DummyModel, self).__init__()
        self.metadata_dim = 6
        self.z_dim = 6
        self.image_size = 28
        self.metadata = None
        self.images = None
        self.loss_name = "PlaceholderLoss"
        self.linear = torch.nn.Linear(1, 1)

    @classmethod
    def from_config(cls, model_config):
        return cls()

    def sample(self, means, logvars):
        # ignore logvars
        # take each mean and return the image corresponding to it
        # assume mean is in self.metadata
        return means

    def encode(self, inputs):

        def is_in(tensor):
            for image in self.images:
                if torch.equal(image, tensor):
                    return True
            return False

        def index_of(tensor):
            # assumes is_in(tensor) == True
            for i, image in enumerate(self.images):
                if torch.equal(image, tensor):
                    return i
            return -1

        new_inds = []
        input_new_ind = []
        # save the inputs in image_index
        for i, input_val in enumerate(inputs):
            input_val *= (1/255)
            if self.images is None:
                self.images = input_val
                continue
            if not is_in(input_val):
                self.images = torch.cat((self.images, input_val))
                new_inds.append(len(self.images) - 1)
                input_new_ind.append(i)
        # get the measurements of the new images
        image_size = inputs.shape[1]
        new_metadata = torch.zeros((self.images.shape[0], self.metadata_dim)).float()
        measurements = torch.Tensor(measure_batch(self.images[new_inds].detach().cpu().numpy()).to_numpy())
        print(measurements)
        measurements = normalize_data(measurements)
        if self.metadata is None:
            self.metadata = measurements
        else:
            new_metadata[:self.metadata.shape[0]] = self.metadata
            new_metadata[new_inds, :] = measurements[input_new_ind]
            self.metadata = new_metadata

        return measurements, torch.Tensor()

    def decode(self, means):
        
        def index_of(tensor):
            # assumes is_in(tensor) == True
            for i, meta in enumerate(self.metadata):
                if torch.equal(meta, tensor):
                    return i
            return -1

        im_indices = []
        for mean in means:
            im_indices.append(index_of(mean))

        return self.images[im_indices]

