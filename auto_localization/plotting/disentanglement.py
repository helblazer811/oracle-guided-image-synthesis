import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
sys.path.append("../..")
import auto_localization.plotting.image_sampling as image_sampling
from auto_localization.plotting.util import * 
from auto_localization.training.training_test import *
from datasets.morpho_mnist.dataset import ImageDataset
from sklearn.manifold import TSNE
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
from scipy.stats import entropy
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import seaborn as sns
import wandb
import torch

"""
    Plots a 2D colored feature embedding based on the intensity of the metadata
    feature with feature_index
"""
def plot_feature_embedding(model, data_manager, feature_index=0, num_samples=500):
    # get a random sample of images from the test_images
    test_images = data_manager.image_test[0:num_samples]
    test_images = test_images.permute(1, 0, 2, 3)
    # run a forward pass on those images
    mean, logvar, encoded_vector, reconstructed = model.forward(test_images.cuda())
    embedding_locations = mean.detach().cpu().numpy()
    # get the metadata characteristics of them
    test_images = test_images.squeeze()
    morpho_mnist_metadata = data_manager.triplet_train.metadata_dataset.measure_images(test_images)
    # package them up and return
    feature_names = ["area", "length", "thickness", "slant", "width", "height"]
    # normalize metadata
    original_morpho_mnist_metadata = np.array(morpho_mnist_metadata)
    morpho_mnist_metadata = original_morpho_mnist_metadata[:, feature_index]
    normalized_metadata = (morpho_mnist_metadata - np.min(morpho_mnist_metadata))/(np.max(morpho_mnist_metadata) - np.min(morpho_mnist_metadata))
    # get line with best spearman correlation
    spearman_direction, spearman = line_with_best_ordering(embedding_locations, original_morpho_mnist_metadata, feature_index)
    # get cmap and colors
    fig = plt.figure(figsize=(6, 5), dpi=500)
    plt.scatter(embedding_locations[:, 0], embedding_locations[:, 1], c=normalized_metadata, s=5, cmap="plasma")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    cbar = plt.colorbar()#matplotlib.cm.ScalarMappable(norm=None, cmap=cmap), pad=0.1)
    cbar.set_label(f"'{feature_names[feature_index]}' Level")
    # plot line with best ordering
    mean = np.mean(embedding_locations, axis=0)
    vector_scale = 3.0
    if not spearman_direction is None:
        quiver = plt.quiver(mean[0], mean[1], spearman_direction[0], spearman_direction[1], scale=vector_scale, angles="xy", zorder=5, pivot="tail")
    # make arrow in legend
    def make_legend_arrow(legend, orig_handle,
                          xdescent, ydescent,
                          width, height, fontsize):
        p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
        return p

    black_circle = Line2D([0], [0], color='black', label='Circle',
                        markerfacecolor='r', markersize=2),
    arrow = plt.arrow(0, 0, 0, 0, label='My label', color='#ffffff00')
    legend = plt.legend([arrow, black_circle], 
                        ['Best Ordered Axis', "Spearman's = {:.3f}".format(spearman)],
                        handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)},
                        bbox_to_anchor=(0.39, 0., 0.2, 0.5))
    
    # save the model
    np_vals = plot_to_numpy(fig)
    wandb.log({f"Colored Feature '{feature_names[feature_index]}' Embedding": wandb.Image(np_vals)})

"""
    Gets latent features and corresponding measurements for a set of images
"""
def get_latents_and_measurements(model, image_dataset, metadata_dataset, num_images=200, metadata_feature=0, latent_feature=0):
    # sample a bunch of images
    images = image_dataset[0:num_images]
    images = torch.Tensor(images) 
    images = images.permute(1, 0, 2, 3)
    # reconstruct the images
    means, _, _, reconstructed_images = model.forward(images.to("cuda"))
    latent_features = means[:, latent_feature]
    latent_features = latent_features.detach().cpu().numpy()
    # measure the metadata features
    measurements = []
    for recon_image_index in range(np.shape(reconstructed_images)[0]):
        recon_image = reconstructed_images[recon_image_index]
        recon_image = recon_image.squeeze().detach().cpu().numpy()
        measurement = metadata_dataset.measure_image(recon_image)
        measurement = measurement[metadata_feature]
        measurements.append(measurement)
        
    return latent_features, measurements
            
def train_isotonic_regression(model, image_dataset, metadata_dataset, metadata_feature=0, latent_feature=0):
    latent_features, measurements = get_latents_and_measurements(model, image_dataset, metadata_dataset, num_images=200, metadata_feature=metadata_feature, latent_feature=latent_feature)
    # train the regressor
    regressor = IsotonicRegression(out_of_bounds='clip', increasing="auto")
    regressor.fit(latent_features, measurements)
    return regressor

def calculate_mse(regressor, model, image_test_dataset, metadata_dataset, latent_feature, num_test=100, metadata_feature=0):
    latent_features, measurements = get_latents_and_measurements(model,
                                                                    image_test_dataset, 
                                                                    metadata_dataset, 
                                                                    metadata_feature=metadata_feature,
                                                                    latent_feature=latent_feature)
    # predict values from the latent features
    predicted_vals = regressor.predict(latent_features)
    predicted_vals = np.array(predicted_vals) - measurements
    # calculate the mse between those values and the original
    mse = np.sum(predicted_vals ** 2)
    # return the mse
    return mse

"""
    Calcualate a matrix of spearman coefficients for each of the
    model dimensions w.r.t each of the metadata characteristics
"""
def calculate_spearman(model, image_test_dataset, metadata_dataset, latent_feature, metadata_feature):
    latent_features, measurements = get_latents_and_measurements(model,
                                                                    image_test_dataset, 
                                                                    metadata_dataset, 
                                                                    num_images=100, 
                                                                    metadata_feature=metadata_feature,
                                                                    latent_feature=latent_feature)
    
    spearman_coeff = spearmanr(latent_features, measurements)
    spearman_coeff = spearman_coeff[0]

    return spearman_coeff

def run_regressions(model, image_dataset, metadata_dataset, metadata_features=[2,3], which_digits=[1]):
    regressors = []
    mses = []
    spearmans = []
    # form a matrix of these plots with the correct labels on the axis
    image_test_dataset = ImageDataset(train=False, which_digits=which_digits)
    latent_dim = model.z_dim
    for metadata_feature_index in range(len(metadata_features)):
        row_regressors = []
        row_mses = []
        row_spearmans = []
        metadata_feature = metadata_features[metadata_feature_index]
        for latent_feature in range(0, latent_dim):
            regressor = train_isotonic_regression(model, image_dataset, metadata_dataset, latent_feature=latent_feature, metadata_feature=metadata_feature)
            mse = calculate_mse(regressor, model, image_test_dataset, metadata_dataset, latent_feature, metadata_feature=metadata_feature)
            spearman = calculate_spearman(model, image_test_dataset, metadata_dataset, latent_feature, metadata_feature)
            row_regressors.append(regressor)
            row_mses.append(mse)
            row_spearmans.append(spearman)
        regressors.append(row_regressors)
        mses.append(row_mses)
        spearmans.append(row_spearmans)
    return regressors, mses, spearmans

def plot_regression(model, regressor, mse, spearman, image_test_dataset, metadata_dataset, latent_feature=0, metadata_feature=0, axs=None):
    latent_features, measurements = get_latents_and_measurements(model,
                                                image_test_dataset, 
                                                metadata_dataset, 
                                                100, 
                                                latent_feature=latent_feature,
                                                metadata_feature=metadata_feature)
    # transform x
    features = ['area', 'length', 'thickness', 'slant', 'width', 'height']
    if axs is None:
        # setup figure
        fig, axs = plt.subplots(1, 1, figsize=(1, 1), dpi=500)
    # set title and axis
    axs.set_title(f"Latent Axis {latent_feature}, Metadata Feature {features[metadata_feature]}")
    axs.set_xlabel(f"Latent Axis '{latent_feature}'")
    axs.set_ylabel(f"Metadata Feature '{features[metadata_feature]}'")
    axs.text(0.8, 0.95, f'MSE = {mse:.2f}',
        horizontalalignment='center',
        verticalalignment='center',
        transform=axs.transAxes,
        backgroundcolor="white")
    axs.text(0.8, 0.90, f'Spearman = {spearman:.2f}',
        horizontalalignment='center',
        verticalalignment='center',
        transform=axs.transAxes,
        backgroundcolor="white")
    x_test = np.linspace(min(latent_features), max(latent_features), 1000)
    axs.scatter(latent_features, measurements)
    axs.plot(x_test, regressor.predict(x_test), 'C1-')

def plot_regressions(model, regressors, mses, spearmans, latent_dim, image_dataset, metadata_dataset, metadata_features=[2, 3], which_digits=[]):
    image_test_dataset = ImageDataset(train=False, which_digits=which_digits)
    fig, axs = plt.subplots(latent_dim, len(metadata_features), figsize=(5*len(metadata_features), 5*latent_dim), dpi=200)
    fig.suptitle("Isotonic Regression of Metadata Feature vs Latent Axis", fontsize=16)
    for metadata_feature_index in range(len(metadata_features)):
        metadata_feature = metadata_features[metadata_feature_index]
        for latent_feature in range(0, latent_dim):
            regressor = regressors[metadata_feature_index][latent_feature]
            mse = mses[metadata_feature_index][latent_feature]
            spearman = spearmans[metadata_feature_index][latent_feature]
            plot_regression(model, regressor, mse, spearman, image_test_dataset, metadata_dataset, latent_feature=latent_feature, metadata_feature=metadata_feature, axs=axs[latent_feature, metadata_feature_index])

    np_vals = plot_to_numpy(fig)
    wandb.log({"Isotonic Regression Correlation Plots": wandb.Image(np_vals)})

    return fig


def plot_global_mask_heatmap(global_mask_vector):
    global_mask_vector = global_mask_vector.detach().cpu().numpy()
    print(global_mask_vector.shape)
    latent_dim = np.shape(global_mask_vector)[0]
    global_mask_vector = np.reshape(global_mask_vector, (latent_dim, 1)).T
    # plots the global mask vector
    fig = plt.figure(figsize=(latent_dim, 2), dpi=100)
    yticklabels = []
    xticklabels = list(np.arange(0, latent_dim))
    # plot a heatmap of the values
    ax = sns.heatmap(global_mask_vector, xticklabels=xticklabels, yticklabels=[], vmin=0.0, cmap="plasma")
    ax.set_xlabel("Latent Dimension")
    ax.set_title("Global Mask Vector")
    np_vals = plot_to_numpy(fig)
    wandb.log({"Global Mask Heatmap": wandb.Image(np_vals)})
    return fig
   

def plot_mse_heatmap(mses, metadata_features, latent_dim):
    features = ['area', 'length', 'thickness', 'slant', 'width', 'height']
    # take mses
    mses = np.array(mses)
    # normalize them across latent dimension axis
    total_rows = np.sum(mses, axis=1)[:, None]
    normalized_mses = 1 - (mses/total_rows)
    # make plot
    fig = plt.figure(figsize=(latent_dim, 2*len(metadata_features)), dpi=100)
    xticklabels = list(np.arange(0, latent_dim))
    yticklabels = [features[index] for index in metadata_features]
    # plot a heatmap of the values
    ax = sns.heatmap(normalized_mses, xticklabels=xticklabels, yticklabels=yticklabels, vmin=0.0, vmax=1.0, cmap="plasma")
    ax.set_xlabel("Metadata Feature")
    ax.set_ylabel("Latent Dimension")
    ax.set_title("Relative Latent Dimension Predictivity of Metadata Feature")

    np_vals = plot_to_numpy(fig)
    wandb.log({"Isotonic Regression Heatmap": wandb.Image(np_vals)})

    return fig

def plot_spearman_heatmap(spearmans, metadata_features, latent_dim):
    features = ['area', 'length', 'thickness', 'slant', 'width', 'height']
    # take spearmans
    spearmans = np.abs(np.array(spearmans))
    # calculate the row-wise entropies
    entropies = []
    for row_index in range(np.shape(spearmans)[0]):
        row_values = spearmans[row_index]
        entropy_value = entropy(row_values)
        entropies.append(entropy_value)
    entropies = np.array(entropies)[:, None]
    # max entorpy   
    entropies = entropies * (1/np.log2(latent_dim))
    # plot parameters
    vmin = 0.0
    vmax = 1.0
    yticklabels = [features[index] for index in metadata_features]
    # make plot
    fig, axs = plt.subplots(ncols=3, gridspec_kw=dict(width_ratios=[latent_dim,1,0.2]))
    axs[0].set_yticklabels(yticklabels)#, rotation=90, ha='right')
    sns.heatmap(spearmans, annot=True, yticklabels=yticklabels, cbar=False, ax=axs[0], vmin=vmin, vmax=vmax, cmap="plasma")
    sns.heatmap(entropies, annot=True, yticklabels=False, cbar=False, ax=axs[1], vmin=vmin, vmax=vmax, cmap="plasma")
    fig.colorbar(axs[1].collections[0], cax=axs[2])
    axs[2].set_ylim(0, 1.0)
    # set labels and titles
    plt.ylabel("Metadata Feature")
    axs[0].set_xlabel("Latent Dimension")
    axs[0].set_title("Spearman Coefficients")   
    axs[1].set_title("Normalized Entropy")
    plt.suptitle("Rank Correlation of Latent Embedding")
    # save the plot
    np_vals = plot_to_numpy(fig)
    wandb.log({"Spearman Coefficient Heatmap": wandb.Image(np_vals)})

    return fig

"""
    Plots graphs showing the fit of the regressors,
    plots a heatmap of the fit for each dimension. 
"""
def plot_isotonic_regressions(model, data_manager, component_weighting):
    print("plot_isotonic_regressions")
    metadata_features = np.nonzero(component_weighting)[0]
    which_digits = data_manager.triplet_train.which_digits
    latent_dim = model.z_dim
    # setup data    
    image_dataset = data_manager.image_train
    image_test_dataset = data_manager.image_test
    metadata_dataset = data_manager.triplet_train.metadata_dataset
    # run regressions
    print("run regressions")
    regressors, mses, spearmans = run_regressions(model, image_dataset, metadata_dataset, metadata_features=metadata_features, which_digits=which_digits)
    # generate and save the plots 
    # plot regressions
    print("plot regressions")
    fig = plot_regressions(model, regressors, mses, spearmans, latent_dim, image_dataset, metadata_dataset, metadata_features=metadata_features, which_digits=which_digits)
    # plot mse heatmaps
    print("plot_heatmaps")
    fig = plot_mse_heatmap(mses, metadata_features, latent_dim)
    # plot spearman coefficient heatmap 
    fig = plot_spearman_heatmap(spearmans, metadata_features, latent_dim)

