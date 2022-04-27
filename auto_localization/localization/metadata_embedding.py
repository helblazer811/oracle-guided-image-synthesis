# sys path
import os
import sys
sys.path.append('..')
# Test the active_localization gui
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #cuda is annoying
#from cvxopt_localize import sum_square
from morpho_mnist.triplet_data_manager import MorphoMNISTDataManager
from localization.active_localization import MCMVLocalizer
#from morpho_mnist.metadata_localization import run_localization_rollouts, make_localizers 

from torch.utils.data import Dataset, TensorDataset
#from morpho_mnist.measure import measure_batch
#from morpho_mnist.morphomnist_repo.io import load_idx

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
class MetadataFakeGenModel():

        def __init__(self):
                self.metadata_dim = 6
                self.image_size = 28
                self.metadata = None
                self.images = None

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
                print(self.images.shape)
                measurements = torch.Tensor(measure_batch(self.images[new_inds]).to_numpy())
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

"""
    Dummy model that returns what it takes in to be used 
    for when the localizers want to be used with no images
"""
class MetadataIdentityModel():

    def __init__(self):
        pass    

    def encode(self, inputs):
        return inputs

    def decode(self, means):
        return means
"""
if __name__ == "__main__":
    # setup the dataset
    print("Loading Morpho trainer")
    morpho_data_manager = MorphoMNISTDataManager(128)
    latent_dims = []
    components = [3]
    k = 1.66
    num_queries = 20
    noise_scale = 0.0
    # load generative model
    print("Loading model...")
    gen_model = MetadataFakeGenModel()
    # make a active localizer object that does random localization
    print("Initializing localizer")
    mode = "MCMV"
    all_localizers = []
    for trial in range(1):
        params = {
                "normalization": 1,
                "k-constant": k
        }
        localizer = MCMVLocalizer(ndim=latent_dims[0])
        # initialize localizer
        localizers = make_localizers(latent_dims, 1, morpho_data_manager, params, model=gen_model, localization_type="MCMV", components=components, noise_scale=noise_scale)
        # do localization 
        run_localization_rollouts(localizers, num_queries=num_queries, components=components)
        all_localizers.extend(localizers)

    all_localizers[0].serialize(all_localizers, f"localizations.pkl")
"""
