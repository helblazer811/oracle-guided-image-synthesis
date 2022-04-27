import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import numpy as np

class CS_Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(CS_Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z, c):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which notion of similarity images are compared"""
        embedded_x, masknorm_norm_x, embed_norm_x, tot_embed_norm_x = self.embeddingnet(x, c)
        embedded_y, masknorm_norm_y, embed_norm_y, tot_embed_norm_y = self.embeddingnet(y, c)
        embedded_z, masknorm_norm_z, embed_norm_z, tot_embed_norm_z = self.embeddingnet(z, c)
        mask_norm = (masknorm_norm_x + masknorm_norm_y + masknorm_norm_z) / 3
        embed_norm = (embed_norm_x + embed_norm_y + embed_norm_z) / 3
        mask_embed_norm = (tot_embed_norm_x + tot_embed_norm_y + tot_embed_norm_z) / 3
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, mask_norm, embed_norm, mask_embed_norm

class ConditionalSimNet(nn.Module):
    def __init__(self, embeddingnet, n_conditions, embedding_size, learnedmask=True, prein=False):
        """ embeddingnet: The network that projects the inputs into an embedding of embedding_size
            n_conditions: Integer defining number of different similarity notions
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            learnedmask: Boolean indicating whether masks are learned or fixed
            prein: Boolean indicating whether masks are initialized in equally sized disjoint 
                sections or random otherwise"""
        super(ConditionalSimNet, self).__init__()
        self.learnedmask = learnedmask
        self.embeddingnet = embeddingnet
        self.n_conditions = n_conditions
        # create the mask
        if learnedmask:
            if prein:
                # define masks 
                self.masks = torch.nn.Embedding(n_conditions, embedding_size)
                # initialize masks
                mask_array = np.zeros([n_conditions, embedding_size])
                mask_array.fill(0.1)
                mask_len = int(embedding_size / n_conditions)
                for i in range(n_conditions):
                    mask_array[i, i*mask_len:(i+1)*mask_len] = 1
                # no gradients for the masks
                self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
            else:
                # define masks with gradients
                self.masks = torch.nn.Embedding(n_conditions, embedding_size)
                # initialize weights
                self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005
        else:
            # define masks 
            self.masks = torch.nn.Embedding(n_conditions, embedding_size)
            # initialize masks
            mask_array = torch.zeros([n_conditions, embedding_size])
            mask_len = int(embedding_size / n_conditions)
            for i in range(n_conditions):
                mask_array[i, i*mask_len:(i+1)*mask_len] = 1
            # no gradients for the masks
            self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)

    def forward(self, x, c):
        embedded_x = self.embeddingnet(x)
        c = c % self.n_conditions
        self.mask = self.masks(c)
        if self.learnedmask:
            self.mask = torch.nn.functional.relu(self.mask)
        masked_embedding = embedded_x * self.mask
        return masked_embedding, self.mask.norm(1), embedded_x.norm(2), masked_embedding.norm(2)

__all__ = ['ResNet', 'cesnet18']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, embedding_size=64, in_channels=3):
        self.in_channels = in_channels
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_embed = nn.Linear(256 * block.expansion, embedding_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_embed(x)

        return x

def resnet18(pretrained=False, in_channels=3, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2], in_channels=in_channels, **kwargs)
    if pretrained:
        state = model.state_dict()
        loaded_state_dict = model_zoo.load_url(model_urls['resnet18'])
        for k in loaded_state_dict:
            if k in state:
                state[k] = loaded_state_dict[k]
        model.load_state_dict(state)
    return model

class ConditionalSimilarityNetworkLoss(nn.Module):

    def __init__(self, tnet=None, triplet_margin=0.1, embed_loss=1.0, mask_loss=1.0):
        super(ConditionalSimilarityNetworkLoss, self).__init__()
        self.empty_linear = nn.Linear(1, 1)
        self.tnet = tnet
        self.kl_beta = 0.0 # backwards compatability
        self.triplet_beta = 0.0
        self.embed_loss = embed_loss
        self.mask_loss = mask_loss
        self.triplet_margin = triplet_margin
        self.triplet_loss_criterion = nn.MarginRankingLoss(margin = self.triplet_margin)

    def triplet_percentage(self, anchor, positive, negative):
        # calculate distances
        distance_anchor_positive = torch.norm(anchor - positive, dim=-1)
        distance_anchor_negative = torch.norm(anchor - negative, dim=-1)
        # test if it is negative
        num_closer = torch.sum((distance_anchor_positive < distance_anchor_negative).int())
        percentage = torch.Tensor([num_closer/anchor.shape[0]])

        return percentage

    def forward(self, real_data, fake_data, mean, logvar, triplet_data, **kwargs):
        if len(triplet_data) == 4:
            anchor, positive, negative, attribute_index = triplet_data
        else:
            anchor, positive, negative = triplet_data
        if not "triplet_input_data" in kwargs:
            raise Exception("triplet input data not passed to loss") 
        triplet_input_data = kwargs["triplet_input_data"]
        # unpack the embedded triplets
        anchor, _ = anchor
        positive, _ = positive
        negative, _ = negative
        # compute auxilery loss
        triplet_percentage = self.triplet_percentage(anchor, positive, negative)
        # compute output
        anchor_in, positive_in, negative_in, attribute_index  = triplet_input_data
        anchor_in = anchor_in.cuda()
        negative_in = negative_in.cuda()
        positive_in = positive_in.cuda()
        attribute_index = attribute_index.cuda()
        dista, distb, mask_norm, embed_norm, mask_embed_norm = self.tnet(anchor_in, negative_in, positive_in, attribute_index)
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(1)
        target = target.cuda()

        loss_triplet = self.triplet_loss_criterion(dista, distb, target)
        loss_embedd = embed_norm / np.sqrt(anchor.size(0))
        loss_mask = mask_norm / anchor.size(0)
        final_loss = loss_triplet + self.embed_loss * loss_embedd + self.mask_loss * loss_mask

        loss_dict = {
            "loss": final_loss,
            "triplet_loss": loss_triplet,
            "mask_loss": loss_mask,
            "embed_loss": loss_embedd,
            "triplet_percentage": triplet_percentage,
        }

        return loss_dict

class ConditionalSimilarityNetwork(nn.Module):

    def __init__(self, latent_dim, in_shape, channels=1, learned_mask=False, prein=False, conditions=[0, 1, 2, 3, 4, 5], triplet_margin=1.0, embed_loss=1.0, mask_loss=1.0):
        super(ConditionalSimilarityNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.in_shape = in_shape
        self.prein = prein
        self.learned_mask = learned_mask
        self.channels = channels
        self.prein = prein
        self.triplet_margin = triplet_margin
        self.embed_loss = embed_loss
        self.mask_loss = mask_loss
        self.conditions = conditions
        # run model setup
        self._setup_model()
        self._setup_loss_function()

    def _setup_loss_function(self):
        self.loss_function = ConditionalSimilarityNetworkLoss(tnet=self.tnet, triplet_margin=self.triplet_margin, embed_loss=self.embed_loss, mask_loss=self.mask_loss)

    @classmethod
    def from_config(cls, config):
        return cls(
                latent_dim = config["latent_dim"],
                in_shape = config["in_shape"],
                channels = config["channels"],
                prein = False if not "prein" in config else config["prein"], 
                learned_mask = False if not "learned_mask" in config else config["learned_mask"],
                triplet_margin = 0.2 if not "triplet_margin" in config else config["triplet_margin"],
                mask_loss = config["mask_loss"],
                embed_loss = config["embed_loss"],
                conditions = config["conditions"],
        )

    def _setup_model(self):
        self.embedding_model = resnet18(pretrained=True, embedding_size=self.latent_dim, in_channels=self.channels)
        self.csn_model = ConditionalSimNet(self.embedding_model, n_conditions=len(self.conditions), 
            embedding_size=self.latent_dim, learnedmask=self.learned_mask, prein=self.prein)
        global mask_var
        mask_var = self.csn_model.masks.weight
        self.tnet = CS_Tripletnet(self.csn_model)
        self.tnet.cuda()
    
    """
        Necessary to fit into existing codebase
    """
    def encode(self, x):
        if len(x.shape) < 3:
            print(x.shape)
            x = x.unsqueeze(0)
        if len(x.shape) < 4:
            x = x.unsqueeze(0)

        embedding_val = self.embedding_model(x)

        return embedding_val, embedding_val

    """
        Necessary to fit into existing codebase.
    """
    def decode(self, x):
        
        return x

    def forward(self, x):
        embedding_val, _ = self.encode(x)
        if len(embedding_val.shape) > 4:
            embedding_val = embedding_val.squeeze()
        return  embedding_val, embedding_val, embedding_val, x

