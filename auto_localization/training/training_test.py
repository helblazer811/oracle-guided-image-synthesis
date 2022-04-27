import sys
sys.path.append("../../")
from auto_localization.localization.localizers.active_localization import pair2hyperplane, KNormalizationType
from auto_localization.models.loss.vae_triplet_loss import VAETripletLoss
from auto_localization.models.MaskedVAE import MaskedVAE
from tqdm import tqdm
import torch
from scipy import stats
import numpy as np
import wandb
from auto_localization.localization.noise_model_selector import NoiseModelSelector
from auto_localization.plotting.util import * 
import matplotlib.pyplot as plt

"""
    Test and measure the percentage of queries that 
    were satisfied. 
"""
def percentage_of_queries_satisfied(model, triplet_dataset):
    # go through the triplet dataset
    pass
    # for each triplet see if it is satisfied by the space

    # return the metric

def response_model_probability(model, triplet_dataset, use_basic_setting=False, train=False):
    model.eval()
    # Do logistic
    noise_model_selector = NoiseModelSelector(model, triplet_dataset=triplet_dataset, localizer_type="RandomLogistic")
    # evaluate the triplets
    triplets = noise_model_selector.evaluate_triplets()
    # evaluate the logistic response model
    if use_basic_setting:
        best_setting = {"k": 1.0, "normalization": 0}
    else:
        best_setting = noise_model_selector.perform_brute_force(triplets)
    logistic_probs = noise_model_selector.compute_success_probabilities(best_setting, triplets)
    logistic_probs = np.stack(logistic_probs).squeeze()
    average = np.mean(logistic_probs)
    wandb.log({f"Logistic Response Model Probability": average})
    # Do triplet
    noise_model_selector = NoiseModelSelector(model, triplet_dataset=triplet_dataset, localizer_type="RandomTriplet")
    # evaluate the triplets
    triplets = noise_model_selector.evaluate_triplets()
    # evaluate the logistic response model
    if use_basic_setting:
        best_setting = {"k": 1.0, "normalization": 0}
    else:
        best_setting = noise_model_selector.perform_brute_force(triplets)
    logistic_probs = noise_model_selector.compute_success_probabilities(best_setting, triplets)
    logistic_probs = np.stack(logistic_probs).squeeze()
    average = np.mean(logistic_probs)
    wandb.log({f"Triplet Response Model Probability": average})


def reconstruction_of_metadata(model, image_dataset, metadata_dataset, num_sample=200):
    # get random indices
    indices = np.random.choice(np.arange(0, len(image_dataset)), size=num_sample)
    # get the metadata 
    true_metadata = metadata_dataset[indices]
    # go through and reconstruct a bunch of images
    images = image_dataset[indices]
    if len(images) == 4:
        images = images.permute(1, 0, 2, 3)
    mean, logvar, z, reconstruction = model.forward(images.cuda())
    reconstruction = reconstruction.detach().cpu().numpy()
    # calculate the metadata for those
    recon_metadata = metadata_dataset.measure_images(reconstruction.squeeze())
    metadata_diff = (recon_metadata - true_metadata)
    difference = np.mean(np.linalg.norm(metadata_diff, axis=1)**2)  

    """
    # get the image test dataset
    image_dataset = triplet_dataset.image_dataset
    # get the metadata test dataset
    try:
        metadata_dataset = triplet_dataset.metadata_dataset
        # get the component_weighting 
        component_weighting = triplet_dataset.oracle.component_weighting
    except:
        return
    # get random indices
    indices = np.random.choice(np.arange(0, len(image_dataset)), size=num_sample)
    # get the metadata 
    true_metadata = metadata_dataset[indices]
    # go through and reconstruct a bunch of images
    images = image_dataset[indices]
    images = images.permute(1, 0, 2, 3)
    mean, logvar, z, reconstruction = model.forward(images.cuda())
    reconstruction = reconstruction.detach().cpu().numpy()
    # calculate the metadata for those
    recon_metadata = metadata_dataset.measure_images(reconstruction.squeeze())
    # calculate the difference in metadata for the input vs output images 
    metadata_diff = (recon_metadata - true_metadata) * component_weighting
    difference = np.mean(np.linalg.norm(metadata_diff, axis=1)**2)  
    """
    # log with wandb
    wandb.log({"metadata_reconstruction_error": difference}) 

"""
    function takes in a set of embedding_locations, morpho_mnist_metadata, and a feature inex
    and figures out which line in the embedding locations space has the best 
"""
def line_with_best_ordering(embedding_locations, morpho_mnist_metadata, feature_index, num_lines=1000):
    # generate a list of 'num_lines' unit vectors sampled from an n-d Gaussian 
    dimensionality = np.shape(embedding_locations)[-1]
    gaussian_random_vectors = np.random.normal(size=(num_lines, dimensionality))
    unit_vectors = gaussian_random_vectors / np.linalg.norm(gaussian_random_vectors, axis=1)[:, None]
    # depracated 2D variant 
    # thetas = np.linspace(0.0, 2*np.pi, num=num_lines)
    # x_vals = np.cos(thetas)
    # y_vals = np.sin(thetas)
    # unit_vectors = np.stack((x_vals, y_vals)).T
    # these vectors will be centered at the mean of a distribution
    mean = np.mean(embedding_locations, axis=0)
    embedding_locations = embedding_locations - mean
    morpho_mnist_metadata = np.array(morpho_mnist_metadata)
    # go thorugh each unit vector
    spearmans = []
    for unit_vector in unit_vectors:
        # project the embedding points down to this line and order them
        overlap_with = np.dot(embedding_locations, unit_vector[:, None])
        # find the spearman rank coefficient of each of the lines
        spearman = stats.spearmanr(overlap_with, morpho_mnist_metadata[:, feature_index])
        spearmans.append(spearman[0])
    spearmans = np.abs(spearmans)
    # return the best line
    best_index = np.argmax(spearmans)
    return unit_vectors[best_index], spearmans[best_index]

def get_morpho_mnist_spearmans(model, data_manager, num_samples=300):
    feature_names = ["area", "length", "thickness", "slant", "width", "height"]
    # get a random sample of images from the test_images
    test_images = data_manager.image_test[0:num_samples]
    test_images = test_images.permute(1, 0, 2, 3)
    # run a forward pass on those images
    mean, logvar, encoded_vector, reconstructed = model.forward(test_images.cuda())
    embedding_locations = mean.detach().cpu().numpy()
    # get the metadata characteristics of them
    test_images = test_images.squeeze()
    morpho_mnist_metadata = data_manager.triplet_train.metadata_dataset.measure_images(test_images)
    # go through each morpho_mnist feature
    num_features = 6
    for feature_index in range(num_features):
        unit_vector, spearmans = line_with_best_ordering(embedding_locations, morpho_mnist_metadata, feature_index)        
        wandb.log({f"feature_{feature_names[feature_index]}": spearmans})

"""
    Measures the probability of all the points in the embedding
    being consistent with a set of queries. I think this might be a 
    more robust metric then percentage of queries that are satisfied. 

"""
def probability_of_query_consistency(model, data_manager, num_queries_testing=100):
    # take the model and get the current embedding valeus
    embedding = data_manager.get_latent_point_training(model, inputs=data_manager.image_test)
    metadata_dataset = data_manager.triplet_test.metadata_dataset
    # get the metadata values corresponding to those embedding values
    # go through num_queries_testing number of queries
    for query_num in tqdm(range(num_queries_testing)):
        # randomly generate a query  
        embedding_index_a = np.random.randint(np.shape(embedding)[0])
        embedding_index_b = np.random.randint(np.shape(embedding)[0])
        a_metadata = metadata_dataset[embedding_index_a] 
        b_metadata = metadata_dataset[embedding_index_b]
        # get query hyperplane
        a, tau = pair2hyperplane([embedding_index_a, embedding_index_b], embedding, KNormalizationType.NORMALIZED)
        # go through each point and see if it is consistent
        total = 0
        num_consistent = 0
        for i in range(len(metadata_dataset)):
            embedding_val = embedding[i]
            metadata_val = np.array(metadata_dataset[i])[3]
            # see if point is on side
            dot_prod = np.dot(a, embedding_val - a*tau)
            positive = dot_prod > 0
            # see which query metadata_val is closer
            dist_a = np.linalg.norm(a_metadata - metadata_val)
            dist_b = np.linalg.norm(b_metadata - metadata_val)
            closer_metadata = 0 if dist_a < dist_b else 1
            # metadata is closer 
            is_consistent = closer_metadata == 0 and positive or closer_metadata == 1 and not positive
            total += 1
            if is_consistent:
                num_consistent += 1
        percent_consistent.append(num_consistent/total)

    percent_consistent = np.array(percent_consistent)
    # calculate the percentage of queries that are consistent
    average_percent_conistent = np.mean(percent_consistent)
    # return the metric
    return average_percent_consistent

def triplet_loss_exclude_dim(model, triplet_dataset, dimension=0):
    triplet_loss_obj = model.loss_function
    if hasattr(model, 'similarity_mode'):
        model.similarity_mode = False
    if hasattr(triplet_dataset, 'indexed'):
        model.loss_function.train_mode = False
        indexed = True
    else:
        indexed = False 
    batch_size = 64
    device = "cuda"
    # make data loader
    triplet_loader = torch.utils.data.DataLoader(triplet_dataset, batch_size=batch_size, shuffle=False)
    # make mask
    latent_dim = model.z_dim
    dim_range = np.arange(0, latent_dim)
    if dimension != -1:
        dim_range = np.ma.array(dim_range, mask=False)
        dim_range.mask[dimension] = True    
        dim_range = dim_range.compressed()
    dim_range = torch.LongTensor(dim_range).to(device)
    # loop will iterate until the shorter of the two iterators is exausted
    triplet_iter = iter(triplet_loader)
    triplet_losses = []
    for triplet_data in tqdm(triplet_iter):
        # load triplet data
        triplet, _ = triplet_data
        if len(triplet) == 4:
            anchor_x, positive_x, negative_x, attribute_index = triplet
        else:
            anchor_x, positive_x, negative_x = triplet
        # put on gpu
        anchor_x = anchor_x.to(device)
        positive_x = positive_x.to(device)
        negative_x = negative_x.to(device)
        # run forward passes on the triplet data
        anchor_mean, anchor_logvar, _, _ = model(anchor_x)
        anchor_mean_excluded = anchor_mean.index_select(1, dim_range)
        positive_mean, positive_logvar, _, _ = model(positive_x)
        positive_mean_excluded = positive_mean.index_select(1, dim_range)
        negative_mean, negative_logvar, _, _ = model(negative_x)
        negative_mean_excluded = negative_mean.index_select(1, dim_range)
        # calculate triplet loss
        triplet_loss_without = triplet_loss_obj.triplet_loss((anchor_mean_excluded, positive_mean_excluded, negative_mean_excluded, _))
        triplet_loss_without = triplet_loss_without.detach().cpu().numpy()
        #calculate difference 
        triplet_losses.append(triplet_loss_without)
    # calculate mean loss
    mean_loss = np.mean(triplet_losses)
    return mean_loss 

"""
    Computes the importance of each of the latent units
    of a model to satisfying each of the triplets. The
    input is a model and a triplet dataset, and the output
    is a vector that sums to one of length latent_dim that 
    corresponds to how important each latent unit is to satisfying 
    triplets. This metric needs to be put into context by showing 
    the percentage of triplets satisfied. 
"""
def compute_triplet_importance(model, triplet_dataset, loss_iters=25):
    device = "cuda"
    # define loss function
    latent_dim = model.z_dim
    # losses for each dim
    losses = np.zeros(latent_dim)
    # calculate triplet loss using whole space
    whole_space_triplet_loss = triplet_loss_exclude_dim(model, triplet_dataset, dimension=-1)
    # calculate cumulative loss
    for dimension in range(0, latent_dim):
        dimension_losses = []
        for iter_num in range(0, loss_iters):
            dimension_loss = triplet_loss_exclude_dim(model, triplet_dataset, dimension=dimension) 
            dimension_losses.append(dimension_loss)
        losses[dimension] = np.mean(dimension_losses)  
    # normalize the loss
    normalized = losses - whole_space_triplet_loss
    normalized /= np.sum(np.abs(normalized))
    return normalized
    
