# sys path
import sys
sys.path.append('../../')
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
import scipy.stats
import wandb
from datasets.morpho_mnist.measure import measure_image, measure_batch
from auto_localization.plotting.util import * 
import matplotlib.gridspec as gridspec
import seaborn as sns
import sklearn.manifold
import scipy.spatial
import cv2

"""
    Takes a bunch of images and generates reconstructions of them, 
    comparing them side by side. 
"""
def plot_reconstructive_sampling(model, images):
    if len(images) <= 1:
        return None
    model.eval()
    fig, axs = plt.subplots(nrows=len(images), ncols=2, figsize=(4, 2*len(images)))
    plt.title("Reconstruction Sampling")
    for i, image in enumerate(images):
        # add column labels for first image
        if i == 0:
            axs[0, 0].set_title("Ground Truth")
            axs[0, 1].set_title("Reconstruction")
        # show left gallery as the ground truth images
        base_image = image.squeeze()
        if len(base_image.shape) > 2:
            base_image = base_image.permute(1, 2, 0) 
        base_image = base_image.detach().numpy()
        axs[i, 0].imshow(base_image)
        # show the right gallery as the reconstructions
        _, _, _, reconstructed = model.forward(image.to("cuda"))
        reconstructed = reconstructed.squeeze().squeeze()
        if len(base_image.shape) > 2:
            reconstructed = reconstructed.permute(1, 2, 0)
        reconstructed = reconstructed.detach().cpu().numpy()
        axs[i, 1].imshow(reconstructed)        

    np_vals = plot_to_numpy(fig)
    wandb.log({"Reconstructive Sampling": wandb.Image(np_vals)})

    return fig


"""
    Takes a bunch of images and generates reconstructions of them, 
    comparing them side by side. 
"""
def plot_similarity_reconstructive_sampling(model, images):
    if len(images) <= 1:
        return None
    model.similarity_mode = True
    model.eval()
    fig, axs = plt.subplots(nrows=len(images), ncols=2, figsize=(4, 2*len(images)))
    plt.title("Similarity Reconstruction Sampling")
    for i, image in enumerate(images):
        # add column labels for first image
        if i == 0:
            axs[0, 0].set_title("Ground Truth")
            axs[0, 1].set_title("Reconstruction")
        # show left gallery as the ground truth images
        base_image = image.squeeze()
        if len(base_image.shape) > 2:
            base_image = base_image.permute(1, 2, 0) 
        base_image = base_image.detach().numpy()
        axs[i, 0].imshow(base_image)
        # show the right gallery as the reconstructions
        _, _, _, reconstructed = model.forward(image.to("cuda"))
        reconstructed = reconstructed.squeeze().squeeze()
        if len(base_image.shape) > 2:
            reconstructed = reconstructed.permute(1, 2, 0)
        reconstructed = reconstructed.detach().cpu().numpy()
        axs[i, 1].imshow(reconstructed)        

    np_vals = plot_to_numpy(fig)
    wandb.log({"Similarity Reconstructive Sampling": wandb.Image(np_vals)})

    return fig

"""
    Plots a grid of images organized based on the tsne embedding of
    the corresponding images
"""
def plot_binned_tsne_grid(tsne_points, embedding, model, num_channels=1, dpi=500, num_x_bins=15, num_y_bins=15, title=""):
    tsne_points = tsne_points.astype(float)
    num_points = np.shape(tsne_points)[0]
    embedding_dim = np.shape(embedding)[1]

    x_min = np.amin(tsne_points.T[0])
    y_min = np.amin(tsne_points.T[1])
    y_max = np.amax(tsne_points.T[1])
    x_max = np.amax(tsne_points.T[0])
    # make the bins from the ranges
    # to keep it square the same width is used for x and y dim
    x_bins, step = np.linspace(x_min, x_max, num_x_bins, retstep=True)
    x_bins = x_bins.astype(float)
    #num_y_bins = np.absolute(np.ceil((y_max - y_min)/step)).astype(float)
    y_bins = np.linspace(y_min, y_max, num_y_bins)
    # sort the tsne_points into a 2d histogram
    hist_obj = scipy.stats.binned_statistic_dd(tsne_points, np.arange(num_points), statistic='count', bins=[x_bins, y_bins], expand_binnumbers=True)
    # sample one point from each bucket
    binnumbers = hist_obj.binnumber
    #num_x_bins = np.amax(binnumbers[0]) + 1
    #num_y_bins = np.amax(binnumbers[1]) + 1
    binnumbers = binnumbers.T
    # some places have no value in a region
    used_mask = np.zeros((num_y_bins, num_x_bins))
    embedding_vals = np.zeros((num_y_bins, num_x_bins, embedding_dim))
    for i, bin_num in enumerate(list(binnumbers)):
        used_mask[bin_num[1], bin_num[0]] = 1
        embedding_vals[bin_num[1], bin_num[0]] = embedding[i]
    model.eval()
    # decode the embedding_vals
    embedding_vals = np.reshape(embedding_vals, (-1, embedding_dim))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_input = torch.tensor(embedding_vals).float()
    images = model.decode(embedding_input.to(device)).detach().cpu().numpy()
    images = np.reshape(images, (num_y_bins , num_x_bins, num_channels, np.shape(images)[2], np.shape(images)[2]))
    images = np.rollaxis(images, 2, 5)
    # plot a grid of the images
    fig, axs = plt.subplots(nrows=np.shape(y_bins)[0], ncols=np.shape(x_bins)[0], constrained_layout=False, dpi=dpi)
    for y in range(num_y_bins):
        for x in range(num_x_bins):
            if used_mask[y, x] > 0.0:
                image = 255-np.uint8(images[y][x].squeeze()*255)
                if num_channels == 3:
                    axs[num_y_bins - 1 - y][x].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                elif num_channels == 1:
                    axs[num_y_bins - 1 - y][x].imshow(image[:, :], cmap="gray")
            axs[y, x].axis('off')

    plt.axis('off')
    mid = (fig.subplotpars.right + fig.subplotpars.left)/2
    plt.suptitle(title, fontsize=12, x=mid)

    return fig

def plot_binned_tsne_images(localizer):
    morpho_data_manager = localizer.data
    model = localizer.gen_model
    embedding = morpho_data_manager.get_latent_point_training(model)
    images = model.decode(torch.Tensor(embedding)).detach().cpu().numpy()
    y = morpho_data_manager.y_train
    # plot the tsne plot
    if np.shape(embedding)[-1] == 2:
        tsne_points = embedding
    else:
        tsne = sklearn.manifold.TSNE(n_components=2, init='pca', perplexity=25)
        tsne_points = tsne.fit_transform(embedding)
    return plot_binned_tsne_grid_ims(tsne_points, images, num_x_bins=15)

"""
    Plots a grid of images organized based on the tsne embedding of
    the corresponding images
"""
def plot_binned_tsne_grid_ims(tsne_points, images, num_x_bins=15):
    tsne_points = tsne_points.astype(float)
    num_points = np.shape(tsne_points)[0]
    x_min = np.amin(tsne_points.T[0])
    y_min = np.amin(tsne_points.T[1])
    y_max = np.amax(tsne_points.T[1])
    x_max = np.amax(tsne_points.T[0])
    # make the bins from the ranges
    # to keep it square the same width is used for x and y dim
    x_bins, step = np.linspace(x_min, x_max, num_x_bins, retstep=True)
    x_bins = x_bins.astype(float)
    num_y_bins = np.absolute(np.ceil((y_max - y_min)/step)).astype(int)
    y_bins = np.linspace(y_min, y_max, num_y_bins)
    # sort the tsne_points into a 2d histogram
    hist_obj = scipy.stats.binned_statistic_dd(tsne_points, np.arange(num_points), statistic='count', bins=[x_bins, y_bins], expand_binnumbers=True)
    # sample one point from each bucket
    binnumbers = hist_obj.binnumber
    num_x_bins = np.amax(binnumbers[0]) + 1
    num_y_bins = np.amax(binnumbers[1]) + 1
    binnumbers = binnumbers.T
    # some places have no value in a region
    used_mask = np.zeros((num_y_bins, num_x_bins))
    image_bins = np.zeros((num_y_bins, num_x_bins, 3, np.shape(images)[2],  np.shape(images)[2]))
    for i, bin_num in enumerate(list(binnumbers)):
        used_mask[bin_num[1], bin_num[0]] = 1
        image_bins[bin_num[1], bin_num[0]] = images[i]
    # plot a grid of the images
    fig, axs = plt.subplots(nrows=np.shape(y_bins)[0], ncols=np.shape(x_bins)[0], constrained_layout=False, dpi=500)
    for y in range(num_y_bins):
        for x in range(num_x_bins):
            if used_mask[y, x] > 0.0:
                image = 255-np.uint8(image_bins[y][x].squeeze()*255)
                image = np.rollaxis(image, 0, 3)
                axs[num_y_bins - 1 - y][x].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axs[y, x].axis('off')

    plt.axis('off')

    return fig

"""
	This plot only works for a 2D embedding
"""
def uniform_image_sample(embedding, model, num_x_bins=15):
    num_points = np.shape(embedding)[0]
    embedding_dim = np.shape(embedding)[1]
    device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get the range of the data
    x_min = np.amin(embedding.T[0])
    y_min = np.amin(embedding.T[1])
    y_max = np.amax(embedding.T[1])
    x_max = np.amax(embedding.T[0])
    # get the number of y_bins 
    # go through each grid tile and get teh embedding location
    x_embeddings = np.linspace(x_min, x_max, num_x_bins)
    y_embeddings = np.linspace(y_min, y_max, num_x_bins)
    xv, yv = np.meshgrid(x_embeddings, y_embeddings)
    embedding_grid = np.array([xv, yv])
    embedding_grid = np.rollaxis(embedding_grid, 0, 3)
    embedding_grid = np.reshape(embedding_grid, (-1, 2))
    # sample the images corresponding to the embedding grid
    embedding_vals = torch.Tensor(embedding_grid).float()
    images = model.decode(embedding_vals.to(device)).detach().cpu().numpy()
    images = images.squeeze()
    images = np.reshape(images, (num_x_bins , num_x_bins,  np.shape(images)[2], np.shape(images)[2]))
    # plot a grid of the images
    fig, axs = plt.subplots(nrows=num_x_bins, ncols=num_x_bins, constrained_layout=True)
    for y in range(num_x_bins):
        for x in range(num_x_bins):
            image = 255-np.uint8(images[y][x].squeeze()*255)
            axs[num_x_bins - 1 - y][x].imshow(image,cmap='gray')
            axs[y, x].axis('off')
    plt.axis('off')
    plt.savefig('test.png', dpi=1000)


"""
	Plots a heatmap of a certain morpho_mnist feature 
	based on a sampling of the latent space
"""
def feature_heatmap(embedding, model, feature="length", title=None, num_x_bins=15):
    num_points = np.shape(embedding)[0]
    embedding_dim = np.shape(embedding)[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plt.tick_params(axis = "both", which = "both", bottom = False, top = False)
    # get the range of the data
    x_min = np.amin(embedding.T[0])
    y_min = np.amin(embedding.T[1])
    y_max = np.amax(embedding.T[1])
    x_max = np.amax(embedding.T[0])
    # go through each grid tile and get teh embedding location
    x_embeddings = np.linspace(x_min, x_max, num_x_bins)
    y_embeddings = np.linspace(y_min, y_max, num_x_bins)
    xv, yv = np.meshgrid(x_embeddings, y_embeddings)
    embedding_grid = np.array([xv, yv])
    embedding_grid = np.rollaxis(embedding_grid, 0, 3)
    embedding_grid = np.reshape(embedding_grid, (-1, 2))
    # sample the images corresponding to the embedding grid
    embedding_vals = torch.Tensor(embedding_grid).float()
    images = model.decode(embedding_vals.to(device)).detach().cpu().numpy()
    images = images.squeeze()
    # measure the 'feature' of these images
    measurements = measure_batch(images)[feature].to_numpy()
    # reshape the measurements as a grid
    measurements = np.reshape(measurements, (num_x_bins , num_x_bins))
    measurements = np.flip(measurements, axis=0)
    # plot the heatmap
    ax = sns.heatmap(measurements, cbar_kws={'label': 'slant value'})
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    if not title is None:
        ax.set_title(title)
    else:
        ax.set_title("'{}' embedding heatmap".format(feature))
    ax.set_ylabel("Latent Dim 2")
    ax.set_xlabel("Latent Dim 1")

"""
	Plots "k_nearest" points near a given one
"""
def sample_near_point(images, k_nearest=5):
	measurements = measure_batch(images).to_numpy()
	rand_ints = np.random.randint(len(images), size=20)
	# plot images
	fig, axs = plt.subplots(nrows=20, ncols=k_nearest, figsize=(k_nearest, 20))
	for i, rand_int in enumerate(rand_ints):	
		sort_val = np.linalg.norm(measurements - measurements[rand_int], axis=1)
		neighbors = sort_val.argsort(axis=0)[:k_nearest]

		for j in range(k_nearest):
			image = 255 - np.uint8(images[neighbors[j]].squeeze())
			axs[i][j].imshow(image,cmap='gray')
			axs[i][j].axis('off')

