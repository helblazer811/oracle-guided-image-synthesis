import sys
sys.path.append("../..")
import os
import auto_localization.oracles.oracle as oracle
#from datasets.celeba.dataset import MetadataDataset
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from datasets.celeba.dataset import ImageDataset
from torch.utils.data import TensorDataset, Dataset

class Vgg_face_dag(nn.Module):

    def __init__(self):
        super(Vgg_face_dag, self).__init__()
        self.meta = {'mean': [129.186279296875, 104.76238250732422, 93.59396362304688],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=0, dilation=1, ceil_mode=False)
        self.fc6 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(in_features=4096, out_features=2622, bias=True)

    def upscale_image(self, x):
        # upscale the image
        # out_size = (-1, self.meta["imageSize"][0], self.meta["imageSize"][1], self.meta["imageSize"][2])
        out_size = self.meta["imageSize"][0], self.meta["imageSize"][1]
        upscale = transforms.Resize(out_size)
        x = upscale(x)
        return x

    def forward(self, x0):
        # upscale
        x0 = self.upscale_image(x0)
        x1 = self.conv1_1(x0)
        x2 = self.relu1_1(x1)
        x3 = self.conv1_2(x2)
        x4 = self.relu1_2(x3)
        x5 = self.pool1(x4)
        x6 = self.conv2_1(x5)
        x7 = self.relu2_1(x6)
        x8 = self.conv2_2(x7)
        x9 = self.relu2_2(x8)
        x10 = self.pool2(x9)
        x11 = self.conv3_1(x10)
        x12 = self.relu3_1(x11)
        x13 = self.conv3_2(x12)
        x14 = self.relu3_2(x13)
        x15 = self.conv3_3(x14)
        x16 = self.relu3_3(x15)
        x17 = self.pool3(x16)
        x18 = self.conv4_1(x17)
        x19 = self.relu4_1(x18)
        x20 = self.conv4_2(x19)
        x21 = self.relu4_2(x20)
        x22 = self.conv4_3(x21)
        x23 = self.relu4_3(x22)
        x24 = self.pool4(x23)
        x25 = self.conv5_1(x24)
        x26 = self.relu5_1(x25)
        x27 = self.conv5_2(x26)
        x28 = self.relu5_2(x27)
        x29 = self.conv5_3(x28)
        x30 = self.relu5_3(x29)
        x31_preflatten = self.pool5(x30)
        x31 = x31_preflatten.view(x31_preflatten.size(0), -1)
        x32 = self.fc6(x31)
        x33 = self.relu6(x32)
        x34 = self.dropout6(x33)
        x35 = self.fc7(x34)
        x36 = self.relu7(x35)
        x37 = self.dropout7(x36)
        x38 = self.fc8(x37)
        return x38

def vgg_face_dag(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Vgg_face_dag()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    model.eval()
    return model

class VGGMetadataDataset(TensorDataset):

    def __init__(self, model=None, train=False, requires_grad=True):
        if model is None:
            path = os.path.join(os.environ["LATENT_PATH"], "auto_localization/oracles/vgg_face_dag.pth")
            self.model = vgg_face_dag(weights_path=path).cuda()
        else:
            self.model = model
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
"""
class VGGFaceOracle(oracle.IOracle):
    
    """
        Takes in metadata dataset and image dataset of a certain format
    """
    def __init__(self):
        # load up a pretrained resnet model that works on 32x32 images
        path = os.path.join(os.environ["LATENT_PATH"], "auto_localization/oracles/vgg_face_dag.pth")
        self.model = vgg_face_dag(weights_path=path).cuda()
        self.metadata_dataset = VGGMetadataDataset(model=self.model, requires_grad=False)
        self.indexed = False

    def calculate_weighted_metadata(self, image):
        return self.calculate_image_model(image)

    def calculate_image_model(self, image):
        # take the image size which is probably (32 x 32) or (64 x 64) and
        # resize it to the necessary shape required as input to ResNet
        if isinstance(image, np.ndarray):
            image = torch.Tensor(image)
        if len(image) < 4:
            image = image.unsqueeze(0)
        return self.model(image.cuda())

    def measure_images(self, images):
        return self.calculate_image_model(images)

    def calculate_resnet_distance(self, image_a, image_b, attribute_index=-1):
        if attribute_index == -1:
            a_resnet = self.calculate_image_model(image_a)
            b_resnet = self.calculate_image_model(image_b)
            distance = torch.norm(a_resnet - b_resnet, p=2)
        else:
            a_resnet = self.calculate_image_model(image_a)
            b_resnet = self.calculate_image_model(image_b)
            distance = torch.norm(a_resnet[:, attribute_index] - b_resnet[:, attribute_index], p=2, dim=1)
 
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
        single_feature_triplet = kwargs["single_feature_triplet"] if "single_feature_triplet" in kwargs else False
        attribute_index = kwargs["attribute_index"] if "attribute_index" in kwargs else -1
        # distance between two 
        reference = reference.cuda()
        item_a = item_a.cuda()
        item_b = item_b.cuda()
        distance_a = self.calculate_resnet_distance(reference, item_a, attribute_index=attribute_index)
        distance_b = self.calculate_resnet_distance(reference, item_b, attribute_index=attribute_index)
        # get answer
        answer = 0 if distance_a < distance_b else 1
        return answer

