import wandb
import sys
sys.path.append("../../")
from auto_localization.plotting.util import *
import auto_localization.experiment_management.util as experiment_util
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
from matplotlib.patches import Ellipse
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
import scipy.stats
import sklearn.manifold

def invert_image(image):
    return 255-np.uint8(image.squeeze()*255)

"""
    Plots the posterior means over time
"""
def save_posterior_mean_over_time(model, data_manager, localizers):
    model.eval()
    if len(localizers) == 0:
        return
    num_localizers = len(localizers)
    num_queries = len(localizers[0].posterior_means)
    fig = plt.figure(figsize=(2*num_queries, 2*num_localizers))
    
    for i, localizer in enumerate(localizers):
        for j, mean in enumerate(localizer.posterior_means):
            ax = fig.add_subplot(num_localizers, num_queries + 1, i * (num_queries + 1) + j + 1)
            # decode mean
            mean = torch.Tensor(mean)[None, :]
            decoded_mean = model.decode(mean.to("cuda")).cpu().detach().numpy().squeeze()
            if decoded_mean.shape[0] == 3:
                decoded_mean = np.transpose(decoded_mean, axes=(1, 2, 0))
            ax.imshow(decoded_mean)
        # show reference image
        ax = fig.add_subplot(num_localizers, num_queries + 1, i * (num_queries + 1) + j + 2)
        reference_image = localizer.reference_data
        if reference_image.shape[0] == 3:
            reference_image = np.transpose(reference_image, axes=(1, 2, 0))
        ax.imshow(reference_image)

    # save the axis 
    np_vals = plot_to_numpy(fig)
    wandb.log({"Posterior Means Over Time": wandb.Image(np_vals)})

"""
     Plots the posterior means over time
"""
def save_similarity_posterior_mean_over_time(model, data_manager, localizers):
    model.similarity_mode = True
    model.eval()
    if len(localizers) == 0:
        return
    num_localizers = len(localizers)
    num_queries = len(localizers[0].posterior_means)
    fig = plt.figure(figsize=(2*num_queries, 2*num_localizers))
    
    for i, localizer in enumerate(localizers):
        for j, mean in enumerate(localizer.posterior_means):
            ax = fig.add_subplot(num_localizers, num_queries + 1, i * (num_queries + 1) + j + 1)
            # decode mean
            mean = torch.Tensor(mean)[None, :]
            decoded_mean = model.decode(mean.to("cuda")).cpu().detach().numpy().squeeze()
            if decoded_mean.shape[0] == 3:
                decoded_mean = np.transpose(decoded_mean, axes=(1, 2, 0))
            ax.imshow(decoded_mean)
        # show reference image
        ax = fig.add_subplot(num_localizers, num_queries + 1, i * (num_queries + 1) + j + 2)
        reference_image = localizer.reference_data
        if reference_image.shape[0] == 3:
            reference_image = np.transpose(reference_image, axes=(1, 2, 0))
        ax.imshow(reference_image)

    # save the axis 
    np_vals = plot_to_numpy(fig)
    wandb.log({"Similarity Posterior Means Over Time": wandb.Image(np_vals)})

"""
     Plots the posterior means over time
"""
def save_nearest_neighbor_image_over_time(model, data_manager, localizers, sample_size=500):
    model.similarity_mode = True
    model.eval()
    # sample a bunch of images and calculate their metadata vectors
    image_indices = np.random.choice(len(data_manager.image_test), size=(sample_size))
    similarity_vectors = []
    for index in image_indices:
        image = data_manager.image_test[index].cuda()
        _, _, similarity_vector, _ = model.forward(image)
        similarity_vectors.append(similarity_vector)

    similarity_vectors = torch.stack(similarity_vectors).cuda()

    def get_closest_image(similarity_vector):
        distances = torch.norm(similarity_vector - similarity_vectors, dim=-1)
        closest_index = torch.argmin(distances)
        closest_image = data_manager.image_test[image_indices[closest_index]]
        return closest_image

    if len(localizers) == 0:
        return
    num_localizers = len(localizers)
    num_queries = len(localizers[0].posterior_means)
    fig = plt.figure(figsize=(2*num_queries, 2*num_localizers))
    
    for i, localizer in enumerate(localizers):
        for j, mean in enumerate(localizer.posterior_means):
            ax = fig.add_subplot(num_localizers, num_queries + 1, i * (num_queries + 1) + j + 1)
            # decode mean
            mean = torch.Tensor(mean)[None, :].cuda()
            # get the closest image
            closest_image = get_closest_image(mean).detach().cpu().numpy()
            if np.shape(closest_image)[0] == 3:
                closest_image = np.moveaxis(closest_image, 0, -1)
            ax.imshow(closest_image.squeeze())
        # show reference image
        ax = fig.add_subplot(num_localizers, num_queries + 1, i * (num_queries + 1) + j + 2)
        reference_image = localizer.reference_data
        if reference_image.shape[0] == 3:
            reference_image = np.transpose(reference_image, axes=(1, 2, 0))
        ax.imshow(reference_image)

    # save the axis 
    np_vals = plot_to_numpy(fig)
    wandb.log({"Nearest Neighbor Posterior Mean Images": wandb.Image(np_vals)})

def save_model_ablation(localizer_metrics, model_configs, pivot_keys, num_trials=1):
    # initialize the plot
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].set_title("Metadata Loss Plot")
    ax[0].set_xlabel("Num Queries")
    ax[0].set_ylabel("Loss")
    ax[1].set_title("Latent Loss Plot")
    ax[1].set_xlabel("Num Queries")
    ax[1].set_ylabel("Loss")
    ax[2].set_title("Percentage Loss Plot")
    ax[2].set_xlabel("Num Queries")
    ax[2].set_ylabel("Loss")
    # iterate over config objects
    for config_index, model_config in enumerate(model_configs):
        # unpack the metrics
        average_metrics, variance_metrics = localizer_metrics[config_index]
        metadata_losses = average_metrics["metadata_loss"]
        metadata_variance = variance_metrics["metadata_loss"]
        localization_losses = average_metrics["localization_loss"]
        localization_variance = variance_metrics["localization_loss"]
        reference_percentile_losses = average_metrics["reference_percentiles"]
        reference_percentile_variance = variance_metrics["reference_percentiles"]
        # generate label name from pivot keys and values
        pivot_values = [model_config[pivot_key] for pivot_key in pivot_keys]
        label_name = "".join([pivot_keys[i]+"_"+str(pivot_values[i])+"_" for i in range(len(pivot_values))])
        # plot the mean line and variance
        # make plots for these losses
        metadata_train = np.arange(0, np.shape(metadata_losses)[0])
        ax[0].plot(metadata_train, metadata_losses, label=label_name)
        ax[0].fill_between(metadata_train, metadata_losses-metadata_variance, metadata_losses+metadata_variance, alpha=0.2)
        # plot test on the right axis
        localization_train = np.arange(0, np.shape(localization_losses)[0])
        ax[1].plot(localization_train, localization_losses, label=label_name)
        ax[1].fill_between(localization_train, localization_losses - localization_variance, localization_losses + localization_variance, alpha=0.2)
        # plot reference percentile
        percentile_train = np.arange(0, np.shape(reference_percentile_losses)[0])
        ax[2].set_ylim(0, 100)
        ax[2].plot(percentile_train, reference_percentile_losses, label=label_name)
        ax[2].fill_between(percentile_train, reference_percentile_losses - reference_percentile_variance, reference_percentile_losses + reference_percentile_variance, alpha=0.2)

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    # save plot with wandb
    wandb.log({"Model Config Ablation": wandb.Image(plot_to_numpy(fig))}) 

def save_localization_ablation(localizer_metrics, localizer_configs, pivot_keys, num_trials=1):
    # initialize the plot
    fig, ax = plt.subplots(1, 4, figsize=(45,15))
    ax[0].set_title("Metadata Loss Plot")
    ax[0].set_xlabel("Num Queries")
    ax[0].set_ylabel("Loss")
    ax[1].set_title("Latent Loss Plot")
    ax[1].set_xlabel("Num Queries")
    ax[1].set_ylabel("Loss")
    ax[2].set_title("Latent Covariance Plot")
    ax[2].set_xlabel("Num Queries")
    ax[2].set_ylabel("Loss")
    ax[3].set_title("Nearest Neighbor Metadata Loss Plot")
    ax[3].set_xlabel("Num Queries")
    ax[3].set_ylabel("Loss")

    # map the metrics to the a
    # iterate over config objects
    for config_index, localizer_config in enumerate(localizer_configs):
        # unpack the metrics
        average_metrics, variance_metrics = localizer_metrics[config_index]
        metadata_losses = average_metrics["metadata_loss"]
        metadata_variance = variance_metrics["metadata_loss"]
        localization_losses = average_metrics["localization_loss"]
        localization_variance = variance_metrics["localization_loss"]
        latent_covariance_mean = average_metrics["latent_covariance_determinant"]
        latent_covariance_variance = variance_metrics["latent_covariance_determinant"]
        nn_mean = average_metrics["nearest_neighbor_loss"]
        nn_variance = variance_metrics["nearest_neighbor_loss"]
        # generate label name from pivot keys and values
        pivot_values = [localizer_config[pivot_key] for pivot_key in pivot_keys]
        label_name = "".join([pivot_keys[i]+"_"+str(pivot_values[i])+"_" for i in range(len(pivot_values))])
        # plot the mean line and variance
        # make plots for these losses
        metadata_train = np.arange(0, np.shape(metadata_losses)[0])
        ax[0].plot(metadata_train, metadata_losses, label=label_name)
        ax[0].fill_between(metadata_train, metadata_losses-metadata_variance, metadata_losses+metadata_variance, alpha=0.2)
        # plot test on the right axis
        localization_train = np.arange(0, np.shape(localization_losses)[0])
        ax[1].plot(localization_train, localization_losses, label=label_name)
        ax[1].fill_between(localization_train, localization_losses - localization_variance, localization_losses + localization_variance, alpha=0.2)
        localization_train = np.arange(0, np.shape(localization_losses)[0])
        ax[2].plot(localization_train, latent_covariance_mean, label=label_name)
        ax[2].fill_between(localization_train, latent_covariance_mean - latent_covariance_variance, latent_covariance_mean + latent_covariance_variance, alpha=0.2)
        localization_train = np.arange(0, np.shape(nn_mean)[0])
        ax[3].plot(localization_train, nn_mean, label=label_name)
        ax[3].fill_between(localization_train, nn_mean - nn_variance, nn_mean + nn_variance, alpha=0.2)

    ax[0].legend(bbox_to_anchor=(0.9, -0.3))
    ax[1].legend(bbox_to_anchor=(0.9, -0.3))
    ax[2].legend(bbox_to_anchor=(0.9, -0.3))
    ax[3].legend(bbox_to_anchor=(0.9, -0.3))

    wandb.log({"Metadata Ablation": wandb.Image(plot_to_numpy(fig))}) 

def save_end_localization_model_ablation(metrics_map, localizer_configs, pivot_keys, num_trials=1):
    # initialize the plot
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    ax.set_title(f"Final Metadata Distance for {pivot_keys[0]}")
    ax.set_xlabel(f"{pivot_keys[0]}")
    ax.set_ylabel("Final Metadata Distance")
    # get end localization metadata distance
    end_distance_map = []
    label_names = []
    x_axis_values = []
    # iterate over config objects
    for config_index, config_object in enumerate(localizer_configs):
        end_localization_metadata_distances = []
        this_config_trials = metrics_map[config_index]
        for trial_localization in this_config_trials:
            average_dict, variance_dict = trial_localization
            average_metadata = average_dict["metadata_loss"]
            last_metadata_distance = average_metadata[-1]
            end_localization_metadata_distances.append(last_metadata_distance) 
        end_distance_map.append(end_localization_metadata_distances)
        # generate label name from pivot keys and values
        pivot_values = [config_object[pivot_key] for pivot_key in pivot_keys]
        x_axis_values.append(pivot_values[0])
        label_name = "".join([pivot_keys[i]+"_"+str(pivot_values[i])+"_" for i in range(len(pivot_values))])
        label_names.append(label_name)
    ax.violinplot(end_distance_map, showmeans=True)
    ax.xaxis.set_ticks(np.arange(len(x_axis_values))+1) #set the ticks to be a
    ax.xaxis.set_ticklabels(x_axis_values, rotation="vertical") # change the ticks' names to x
    # save plot with wandb
    wandb.log({"End Localization Ablation": wandb.Image(plot_to_numpy(fig))}) 

"""
    Saves plots of various metdata distance metrics
"""
def save_localization_broken_down_metadata_distance_plots(localizer_metrics, component_weighting):
    feature_names = ["area", "length", "thickness", "slant", "width", "height"]
    # unpack the metrics
    means, variances = localizer_metrics    
    metadata_losses = means["broken_down_metadata_loss"]    
    metadata_variance = variances["broken_down_metadata_loss"]    
    # make plots for these losses
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    metadata_train = np.arange(0, np.shape(metadata_losses)[0])
    ax.set_title("Broken Down Metadata Loss Plot")
    for feature_index in range(np.shape(metadata_losses)[1]):
        weight = component_weighting[feature_index]
        if weight <= 0.0:
            continue
        ax.plot(metadata_train, metadata_losses[:, feature_index], label=feature_names[feature_index])
        ax.fill_between(metadata_train, metadata_losses[:, feature_index]-metadata_variance[:, feature_index], metadata_losses[:, feature_index]+metadata_variance[:, feature_index], alpha=0.2)
    ax.set_ylim(0)
    # axis labels/title
    plt.xlabel("Num Queries")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")

    wandb.log({"broken_down_metadata_plot": wandb.Image(plot_to_numpy(fig))}) 

"""
    Saves plots of various metdata distance metrics
"""
def save_localization_metadata_distance_plots(localizer_metrics):
    print("save localization metadata distance plots")
    # unpack the metrics
    means, variances = localizer_metrics    
    metadata_losses = means["metadata_loss"]    
    localization_losses = means["localization_loss"]    
    metadata_variance = variances["metadata_loss"]    
    localization_variance = variances["localization_loss"] 
    latent_covariance = means["latent_covariance_determinant"]
    nearest_neighbor_losses = means["nearest_neighbor_loss"]
    nearest_neighbor_variance = variances["nearest_neighbor_loss"]
    latent_covariance = np.concatenate((np.array([latent_covariance[0]]), latent_covariance))
    # make plots for these losses
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    metadata_train = np.arange(0, np.shape(metadata_losses)[0])
    ax[0].set_title("Metadata Loss Plot")
    ax[0].plot(metadata_train, metadata_losses, label="Metadata Loss")
    ax[0].fill_between(metadata_train, metadata_losses-metadata_variance, metadata_losses+metadata_variance, alpha=0.2)
    # make plots for the 
    ax[0].plot(metadata_train[:-1], nearest_neighbor_losses, label="Nearest Neighbor Metadata Loss")
    ax[0].fill_between(metadata_train[:-1], nearest_neighbor_losses-nearest_neighbor_variance, nearest_neighbor_losses+nearest_neighbor_variance, alpha=0.2)
    ax[0].set_ylim(0)
    ax[0].legend()
    # plot test on the right axis
    localization_train = np.arange(0, np.shape(localization_losses)[0])
    ax[1].set_title("Latent Loss Plot")
    ax[1].plot(localization_train, localization_losses, label="Latent Loss")
    ax[1].set_ylabel("Latent Loss")
    ax[1].fill_between(localization_train, localization_losses-localization_variance, localization_losses+localization_variance, alpha=0.2)
    # plot secondary axis on the right
    right_ax = ax[1].twinx()
    # make a plot with different y-axis using second axis object
    try:
        right_ax.plot(localization_train[:-1], latent_covariance, color="orange", label="Latent Covariance Determinant")
    except:
        right_ax.plot(localization_train, latent_covariance, color="orange", label="Latent Covariance Determinant")
    right_ax.set_ylabel("Latent Covariance Determinant", color="orange", fontsize=14)
    # axis labels/title
    plt.xlabel("Num Queries")
    plt.legend(loc="upper left")

    wandb.log({"metadata_plot": wandb.Image(plot_to_numpy(fig))}) 

def save_reference_percentile_plots(localizer_metrics, average_localizer_metrics, plot_individual=False):
    # plot the mean in a solid color
    means, variances = average_localizer_metrics    
    reference_percentile_mean = np.array(means["reference_percentiles"])
    reference_percentile_variance = np.array(variances["reference_percentiles"])
    # make plots for these losses
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    num_percentiles = np.arange(0, np.shape(reference_percentile_mean)[0])
    # plot average
    ax.plot(num_percentiles, reference_percentile_mean, color="blue")
    ax.fill_between(num_percentiles, reference_percentile_mean-reference_percentile_variance, reference_percentile_mean+reference_percentile_variance, alpha=0.2)
    # plot test on the right axis
    if plot_individual:
        for metrics_pack in localizer_metrics:
            percentiles = metrics_pack["reference_percentiles"]
            ax.plot(num_percentiles, percentiles, alpha=0.25, color="blue")
    # axis labels/title
    ax.set_title("Reference Percentile Plot")
    ax.set_ylim(0, 100)
    ax.set_xticks(num_percentiles)
    plt.xlabel("Num Queries")
    plt.ylabel("Reference Percentile")
    plt.legend(loc="upper left")

    wandb.log({"Reference Percentile Plot": wandb.Image(plot_to_numpy(fig))}) 

def save_nearest_neighbor_percentile_plots(localizer_metrics, average_localizer_metrics, plot_individual=True):
    # plot the mean in a solid color
    means, variances = average_localizer_metrics    
    reference_percentile_mean = np.array(means["nearest_neighbor_percentile"])
    reference_percentile_variance = np.array(variances["nearest_neighbor_percentile"])
    # make plots for these losses
    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    num_percentiles = np.arange(0, np.shape(reference_percentile_mean)[0])
    # plot average
    ax.plot(num_percentiles, reference_percentile_mean, color="blue")
    ax.fill_between(num_percentiles, reference_percentile_mean-reference_percentile_variance, reference_percentile_mean+reference_percentile_variance, alpha=0.2)
    # plot test on the right axis
    if plot_individual:
        for metrics_pack in localizer_metrics:
            percentiles = metrics_pack["nearest_neighbor_percentile"]
            ax.plot(num_percentiles, percentiles, alpha=0.25, color="blue")
    # axis labels/title
    ax.set_title("Nearest Neighbor Percentiles")
    ax.set_ylim(0, 100)
    ax.set_xticks(num_percentiles)
    plt.xlabel("Num Queries")
    plt.ylabel("Percentile")
    plt.legend(loc="upper left")

    wandb.log({"Nearest Neighbor Percentiles": wandb.Image(plot_to_numpy(fig))}) 


""" 
    Plots the posterior means over time for a given localizer
"""
def plot_posterior_mean_images(model, localizer, data_manager, plot_num=1):
    model.eval()
    estimates = localizer.posterior_means
    reference_image = invert_image(localizer.reference_data)
    queries = localizer.queries
    choices = localizer.choices
    variances = localizer.vars
    z = localizer.embedding
    y = data_manager.image_train.labels
    mode = localizer.mode
    
    def decode_estimates(estimates):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        z = torch.tensor(estimates).float()
        images = model.decode(z.to(device)).detach().cpu().numpy()
        return images

    plt.xlabel("Query Number")
    plt.ylabel("Current Ideal Point")

    # get images
    images = decode_estimates(estimates)
    # make the figure
    width = len(queries) + 1
    height = 2
    fig = plt.figure(figsize = (width*4, height*4))
    gs1 = gridspec.GridSpec(height, width)
    gs1.update(wspace=0.25, hspace=0.05) # set the spacing between axes. 
    # plot images
    for i in range(width - 1):
        ax1 = plt.subplot(gs1[i])
        plt.axis('on')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlabel(i+1)
        ax1.set_aspect('equal')
        image = 255-np.uint8(images[i].squeeze()*255)
        ax1.imshow(image,cmap='gray')

    # plot the decision boundary embedding plots
    for i in range(width - 1):
        ax1 = plt.subplot(gs1[i])
        plt.axis('on')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlabel(i+1)
        ax1.set_aspect('equal')
        image = 255-np.uint8(images[i].squeeze()*255)
        ax1.imshow(image,cmap='gray')

    # plot the decision boundary embedding plots
    for i in range(width - 1):
        grid_index = width + i
        ax1 = plt.subplot(gs1[grid_index])
        ax1.tick_params(labelcolor='none', top=False,
                    bottom=False, left=False, right=False)
        embedding_with_planes(z, y, [queries[i]], [choices[i]], 
            axis=ax1, current_estimate=estimates[i], reference_data=localizer.embedded_reference, variance=variances[i])

    # plot reference image
    ax1 = plt.subplot(gs1[width-1])
    ax1.imshow(reference_image,cmap='gray')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel("Reference Image")

    # plot posterior path embedding
    ax1 = plt.subplot(gs1[width*height - 1])
    embedding_with_posterior_path(z, y, estimates, ax=ax1, reference_data=localizer.embedded_reference)
    # save plot
    wandb.log({"posterior_mean_plot{}".format(plot_num): wandb.Image(plot_to_numpy(fig))}) 

"""
    Helper function for calculating the tsne points and manifold
       returns (embedded_points, tsne_function)
"""
def calculate_tsne(z):
    tsne = sklearn.manifold.TSNE(n_components=2, init='pca', random_state=0)
    tsne_points = tsne.fit_transform(z)

    return tsne_points, tsne.fit_transform

"""        
    Gets a figure with the tsne points plotted
"""
def plot_points(tsne_points, y, zorder=0, axis=None, bounds=None):
    # plot points
    if axis == None:
        ax = plt
    else:
        ax = axis
        #ax.set_aspect(1.0)
    if not bounds is None:
        ax.axis(bounds)
    else:
        ax.axis([np.min(tsne_points[:,0])*1.2, np.max(tsne_points[:,0])*1.2,
              np.min(tsne_points[:,1])*1.2, np.max(tsne_points[:,1])*1.2])
    
    """
    if isinstance(y[0], torch.Tensor):
        for i in range(len(tsne_points)):
            label = str(y[i].item())
            color = plt.cm.Set1(0.1+y[i].item()/12.)
            ax.text(tsne_points[i,0], tsne_points[i,1], label,
                color = color,
                fontdict = {'weight': 'bold', 'size': 9},
                zorder = zorder,
                clip_on=True)
    else:
    """
    ax.set_aspect('equal')
    ax.scatter(tsne_points[:, 0], tsne_points[:, 1], color = "blue")

"""
    This function plots tsne embeddings of data with 
    planes representing the decision boundaries of user queries. 
    There are three possible ways I can think of doing it.

    1. "linear": Calculating a linear bisecting plane in the z dimension and 
    projecting it down to the tsne dimension
    2. "linear_projected": Calculating the linear bisecting plane of the already projected data.
    This would probably look better, but some things will exist on a different 
    side of the hyperplane from where they would in the z dimension because tsne
    does non-linear dimensionality reduction.
    3. "tsne": A plane could be calculated in the z dimension and points along the plane
    in the z-dimension could be sampled and then projected down with t-sne. 
    The points can then be used to draw a boundary(likely curved) in the t-sne dimension.

    This ignores localization and 
"""
def embedding_with_planes(z, y, queries, query_responses, mode="linear_projected", axis=None, current_estimate=None, reference_data=None, variance=None):
    num_points, z_dim = z.shape
    if axis != None:
        ax = axis
    else:
        fig, ax = plt.subplots()

    """
        Calculates the "linear_projected" planes. 
        The plane is represented as a vector inline with the plane
        and a point. This would normally need a normal vector and point,
        but because the space is 2d, and the boundary 1d there is no need. 
    """
    def calculate_linear_projected(queries_projected):
        # print("linear projected")
        lines = []
        # go through each of the queries
        for i, query in enumerate(queries):
            projected_point_a = queries_projected[i][0]
            projected_point_b = queries_projected[i][1]
            # calculate the bisecting planes from these points
            midpoint = (projected_point_a + projected_point_b) * 0.5
            normal_vector = projected_point_a - midpoint
            # calculate cross of normal vector and vector pointing out of the page
            out_of_page_vector = np.array([0, 0, 1])
            normal_vector_3d = np.array([normal_vector[0], normal_vector[1]])
            crossed_vector = np.cross(normal_vector_3d, out_of_page_vector)
            # drop the last dimension of this vector
            tangent_vector = np.array([crossed_vector[0], crossed_vector[1]])
            # add to the list
            lines.append((midpoint, tangent_vector))

        return lines

    """
        Gets the points for a bisecting line in the plot 
        based on a tangent vector and point range
    """
    def calculate_line_points(midpoint, unit_tangent, point_range):
        (x_min, x_max), (y_min, y_max) = point_range
        # get the leftmost point inline with the tangent vector
        scale_factor_min = (x_min -midpoint[0])/unit_tangent[0]
        min_location = midpoint + unit_tangent * scale_factor_min
        # get the rightmost point inline with the tangent vector
        scale_factor_max = (x_max - midpoint[0])/unit_tangent[0]
        max_location = midpoint + unit_tangent * scale_factor_max
        # convert points to plotable form
        x = [min_location[0], max_location[0]]
        y = [min_location[1], max_location[1]]

        return x,y

    """
        Plot the planes on the given figure. 
        The planes are really lines for the case of 2d tsne plots
    """
    def plot_planes(planes, point_range):
        # print("plotting planes")
        for plane in planes:
            midpoint, tangent_vector = plane
            x, y = calculate_line_points(midpoint, tangent_vector, point_range)
            # plot
            ax.plot(x, y)

    def plot_variance_ellipse(points, n_std = 1):
        pearson = variance[0, 1]/np.sqrt(variance[0, 0] * variance[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            color="red",
            alpha=0.5)

        scale_x = np.sqrt(variance[0, 0]) * n_std
        scale_y = np.sqrt(variance[1, 1]) * n_std
        mean = current_estimate

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean[0], mean[1])

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def calculate_point_range(points):
        transposed_points = points.T
        x_max = np.amax(transposed_points[0])
        x_min = np.amin(transposed_points[0])
        y_min = np.amax(transposed_points[1])
        y_max = np.amin(transposed_points[1])

        return (x_min, x_max), (y_min, y_max)

    """
        Highlights the query points
    """
    def plot_query_points(queries_projected):
        # project the query points into tsne
        for i, query in enumerate(queries):
            response = query_responses[i]
            point_a = queries_projected[i][0]
            point_b = queries_projected[i][1]
            response_point = point_a if response == 1 else point_b
            other_point = point_b if response == 1 else point_a
            # plot the point larger then normal
            ax.scatter(response_point[0], response_point[1], c="#00ff00", s=50, zorder=1)
            ax.scatter(other_point[0], other_point[1], c="#ff0000", s=50, zorder=1)

    def plot_current_estimate(current_estimate):
        ax.scatter(current_estimate[0], current_estimate[1], c="#0000ff", s=50, zorder=1)

    def plot_reference_point(reference_data):
        ax.scatter(reference_data[0], reference_data[1], c="#ffff00", s=50, zorder=1, marker="*", label="reference")

    """
        Shades the side of a decision boundary based on the 
        query response
    """
    def plot_shaded_region(planes, point_range):
        """
            Sorts a list of points to make a convex hull
        """
        def make_convex_polygon(points):
            # find the centroid
            centroid = np.mean(points, axis=0)
            sorting_values = [np.arctan2(x - centroid[0], y - centroid[1]) for x,y in points]
            tuple_list = list(zip(sorting_values, points))
            tuple_list = sorted(tuple_list, key=lambda x: x[0])
            vals, sorted_list = zip(*tuple_list) # unzip
            # make a polygon from the points
            polygon = Polygon(sorted_list, True)
            return polygon

        (x_min, x_max), (y_min, y_max) = point_range
        # go through each plane/queries
        for i, response in enumerate(query_responses):
            midpoint, tangent_vector = planes[i]
            # unit tangent vector
            unit_tangent = tangent_vector / np.linalg.norm(tangent_vector)
            x, y = calculate_line_points(midpoint, unit_tangent, point_range)
            # calculate the normal vector from the unit_tangent
            out_of_page_vector = np.array([0, 0, 1])
            tangent_vector_3d = np.array([unit_tangent[0], unit_tangent[1]])
            crossed_vector = np.cross(tangent_vector_3d, out_of_page_vector)
            # drop the last dimension of this vector
            normal_vector = np.array([crossed_vector[0], crossed_vector[1]])
            # make sure the sign of the normal vector is such that it points towards the query_response_vector
            query_response_vector = queries_projected[i][query_responses[i]] - midpoint
            # dot the query_response_vector with another vector on the plane
            other_vector = query_response_vector - np.array([x[0], y[0]])
            # multiply that by the normal_vector
            normal_vector = normal_vector * np.dot(normal_vector, other_vector)
            # dot the query with the normal vector
            response_sign = -1*np.dot(query_response_vector, normal_vector) 
            # also dot the corners
            # top-left -> top-right -> bottom-left -> bottom-right
            top_left = np.dot(np.array([x_min, y_max]), normal_vector)
            top_right = np.dot(np.array([x_max, y_max]), normal_vector)
            bottom_left = np.dot(np.array([x_min, y_min]), normal_vector)
            bottom_right = np.dot(np.array([x_max, y_min]), normal_vector)
            # add the corners with the same sign as the plane to the shading polygon
            polygon_points = []
            if np.sign(top_left) == np.sign(response_sign):
                polygon_points.append([x_min, y_max])
            if np.sign(top_right) == np.sign(response_sign):
                polygon_points.append([x_max, y_max])
            if np.sign(bottom_left) == np.sign(response_sign):
                polygon_points.append([x_min, y_min])
            if np.sign(bottom_right) == np.sign(response_sign):
                polygon_points.append([x_max, y_min])
            # add the min and max points of the plane
            polygon_points.append([x[0], y[0]])
            polygon_points.append([x[1], y[1]])
            polygon_points = np.array(polygon_points)
            # the shape is a convex hull so I may be able to sort the points such that it comes out that way
            polygon = make_convex_polygon(polygon_points)
            # create a patch list and draw it
            patches = [polygon]
            patch_collection = PatchCollection(patches, alpha=0.4)
            ax.add_collection(patch_collection)

    queries_projected = None
    # main code
    if z_dim > 2:
        # include all of the queries in the tsne projection
        num_queries = np.shape(queries)[0]
        points_to_project = np.concatenate((z, np.reshape(queries, (num_queries*2,-1))), axis=0)
        points, tsne_function = calculate_tsne(points_to_project)
        queries_projected = points[-num_queries*2:,:]
        queries_projected = np.reshape(queries_projected, (num_queries, 2, -1))
        points = points[0:-num_queries*2,:] 
    else:
        points = z
        queries_projected = queries
        tsne_function = lambda x: x

    point_range = calculate_point_range(points)
    plot_points(points, y, axis=ax, bounds=(point_range[0][0], point_range[0][1], point_range[1][0], point_range[1][1]))
    planes = calculate_linear_projected(queries_projected)
    plot_planes(planes, point_range)
    plot_variance_ellipse(points)
    plot_query_points(queries_projected)
    # plot_shaded_region(planes, point_range)
    if not reference_data is None:
       plot_reference_point(reference_data)
    if not current_estimate is None:
       plot_current_estimate(current_estimate)

    plt.tick_params(labelcolor='none', top=False,
                   bottom=False, left=False, right=False)
    return ax

def save_embedding_with_posterior_path(model, localizer, image_dataset, num_sample=500):
    print("save embedding with posterior path")
    if hasattr(model, "similarity_mode"):
        model.similarity_mode = True
    # embed a sample of images from the image dataset
    image_sample = image_dataset[0:num_sample]
    image_sample = torch.Tensor(image_sample).cuda()
    points, _, _, _ = model.forward(image_sample.permute(1, 0, 2, 3))
    points = np.array(points.detach().cpu().numpy())
    # make labels
    labels = np.zeros(np.shape(points)[0])
    # get the data from the localizer
    posterior_means = localizer.posterior_means
    embedded_reference = localizer.embedded_reference
    # make the plot    
    fig = embedding_with_posterior_path(points, labels, posterior_means, reference_data=embedded_reference) 
    # save wandb plot
    wandb.log({"Embedding With Posterior Path": wandb.Image(plot_to_numpy(fig))}) 
    if hasattr(model, "similarity_mode"):
        model.similarity_mode = False

"""
    Plots an embedding  with the path of posterior means over time
"""
def embedding_with_posterior_path(z, y, posteriors, ax=None, reference_data=None, bounds=None):
    num_points, z_dim = z.shape
    posterior_dim = np.shape(posteriors)[1]
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    """
        Plots a path of the posterior through the graph
    """
    def plot_posterior_path(posteriors):
        lines = []
        colors = []
        print("plot posterior path")
        for i in range(0, len(posteriors) - 1):
            a, b = posteriors[i], posteriors[i + 1]
            lines.append([a, b])
            colors.append("red")
        line_collection = LineCollection(lines, zorder=2, linewidths=3, colors=colors)
        ax.add_collection(line_collection)

    # main code
    if z_dim > 2:
        tsne_points, tsne_function = calculate_tsne(np.concatenate((z,posteriors,[reference_data])))
        posteriors = tsne_points[-len(posteriors)-1:-1]
        reference_data = tsne_points[-1]
        tsne_points = tsne_points[:-len(posteriors)-1]
    else:
        tsne_points = z
        reference_data = reference_data

    # label start and finish
    ax.annotate("start", (posteriors[0][0], posteriors[0][1]))
    ax.annotate("finish", (posteriors[-1][0], posteriors[-1][1]))

    if not reference_data is None:
        plt.plot(reference_data[0], reference_data[1], marker="*",  markersize=12, color="red")

    plot_points(tsne_points, y, zorder = 1, axis=ax, bounds=bounds)
    plot_posterior_path(posteriors)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    # save the axis 
    #np_vals = plot_to_numpy(fig)
    #andb.log({"Posterior Means Over Time": wandb.Image(np_vals)})
    
    return fig
