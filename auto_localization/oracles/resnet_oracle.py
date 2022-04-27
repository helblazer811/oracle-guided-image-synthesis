import sys
sys.path.append("../..")
import auto_localization.oracles.oracle as oracle
import numpy as np
import torch
import torchvision.models as models
import datasets.celeba.dataset
from datasets.celeba.dataset import ImageDataset
from torch.utils.data import TensorDataset, Dataset

class ResnetMetadataDataset(TensorDataset):

    def __init__(self, train=False, requires_grad=True):
        # load up a pretrained resnet model that works on 32x32 images
        self.model = models.resnet18(pretrained=True).cuda().eval()
        self.train = train
        self.requires_grad = requires_grad
        self.image_dataset = ImageDataset(input_shape=(32, 32), train=train)

    def __getitem__(self, idx):
        image = self.image_dataset[idx].cuda()
        self.model.eval()
        if len(image.shape) < 4:
            image = image.unsqueeze(0)
        if not self.requires_grad:
            with torch.no_grad():
                metadata = self.model(image).detach()
        else:
            metadata = self.model(image)
        return metadata

"""
    This is an oracle that answers queries based on 
    some way of calculating metadata. Each of the metadata
    is assumed to be continuous and normalized.

    TODO handle categorical metadata features
"""
class ResnetOracle(oracle.IOracle):
    
    """
        Takes in metadata dataset and image dataset of a certain format
    """
    def __init__(self, metadata_dataset=None):
        # load up a pretrained resnet model that works on 32x32 images
        self.model = models.resnet18(pretrained=True).cuda().eval()
        if not metadata_dataset is None:
            self.metadata_dataset = metadata_dataset
        else:
            self.metadata_dataset = ResnetMetadataDataset()
        self.indexed = False

    def calculate_weighted_metadata(self, image):
        return self.calculate_image_resnet(image)

    def calculate_image_resnet(self, image):
        # take the image size which is probably (32 x 32) or (64 x 64) and
        # resize it to the necessary shape required as input to ResNet
        if not isinstance(image, torch.Tensor):
            image = torch.Tensor(image)
        image = image.cuda()
        if len(image) < 4:
            image = image.unsqueeze(0)
        return self.model(image)

    def calculate_resnet_distance(self, image_a, image_b):
        a_resnet = self.calculate_image_resnet(image_a)
        b_resnet = self.calculate_image_resnet(image_b)
        distance = torch.norm(a_resnet - b_resnet, p=2)
        return distance

    """
        Returns an answer to a query based on 
        pre-defined query answers.

        queries come in the form of 
        query = (reference, item_a, item_b)
    """
    def answer_query(self, query, **kwargs):
        # unpack query
        reference, item_a, item_b = query
        # distance between two 
        distance_a = self.calculate_resnet_distance(reference, item_a)
        distance_b = self.calculate_resnet_distance(reference, item_b)
        # get answer
        answer = 0 if distance_a < distance_b else 1
        return answer

