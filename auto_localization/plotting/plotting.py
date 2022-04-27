import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
sys.path.append("../..")
import auto_localization.plotting.image_sampling as image_sampling
from auto_localization.plotting.util import * 
from auto_localization.training.training_test import *
from sklearn.manifold import TSNE
import wandb

source_dir = "/home/alec/latent-space-localization/source/morpho_mnist"

"""
    What is the relationship between the latent covariance and whether or 
    not a triplet is satisfied?
"""
def plot_latent_covariance_triplet_satisfied_relationship(model, triplet_dataset, num_sample=200):

    def is_satisfied(triplet_data):
        anchor, positive, negative = triplet_data
        anchor = anchor[0]
        positive = positive[0]
        negative = negative[0]
        # calculate distances
        distance_anchor_positive = torch.norm(anchor - positive, dim=-1)
        distance_anchor_negative = torch.norm(anchor - negative, dim=-1)
        # test if it is negative
        is_closer = (distance_anchor_positive < distance_anchor_negative).int()
        return is_closer

    def compute_mean_determinant(triplet_data):
        anchor_var = triplet_data[0][1].exp().detach().cpu().numpy()
        positive_var = triplet_data[1][1].exp().detach().cpu().numpy()
        negative_var = triplet_data[2][1].exp().detach().cpu().numpy()

        anchor_det = np.prod(anchor_var)
        positive_det = np.prod(positive_var)
        negative_det = np.prod(negative_var)
        
        mean = (anchor_det + positive_det + negative_det) / 3
        return mean

    """
        Performs forward pass on a triplet
    """
    def triplet_forward(triplet):
        triplet, _ = triplet
        anchor_x, positive_x, negative_x, _= triplet
        # put on gpu
        anchor_x = anchor_x.cuda()
        positive_x = positive_x.cuda()
        negative_x = negative_x.cuda()
        # run forward passes on the triplet data
        anchor_mean, anchor_logvar, _, _ = model(anchor_x)
        positive_mean, positive_logvar, _, _ = model(positive_x)
        negative_mean, negative_logvar, _, _ = model(negative_x)

        return ((anchor_mean, anchor_logvar), (positive_mean, positive_logvar), (negative_mean, negative_logvar))

    satisfied_triplet_covariances = []
    unsatisfied_triplet_covariances = []
    # go through a set of triplets
    for triplet_index in range(num_sample):
        triplet = triplet_dataset[triplet_index]
        # forward pass the triplet
        triplet_data = triplet_forward(triplet)
        # compute the average latent covariance determinant
        mean_det = compute_mean_determinant(triplet_data)
        # determine if the triplet is satisfied
        satisfied = is_satisfied(triplet_data)
        #  add the average to the correct list 
        if satisfied:
            satisfied_triplet_covariances.append(mean_det)
        else:
            unsatisfied_triplet_covariances.append(mean_det)
    # plot a violin plot of the two arrays
    fig, ax = plt.subplots()
    ax.violinplot([satisfied_triplet_covariances, unsatisfied_triplet_covariances])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Satisfied', 'Unsatisfied'])
    ax.set_ylabel("Covariance Determinant")
    # log it
    np_vals = plot_to_numpy(fig)
    wandb.log({"latent_covariance_triplet_satisfied_relationship": wandb.Image(np_vals)})


def train_test_loss(train_loss, test_loss, filename=None, plot_name="train_test_loss"):
    fig, ax = plt.subplots()
    x_train = np.arange(0, len(train_loss))
    ax.plot(x_train, train_loss, label="train")
    x_test = np.arange(0, len(test_loss))
    ax.plot(x_test, test_loss, label="test")
    # axis labels/title
    plt.title("Train/Test Loss Plot")
    plt.xlabel("Num Iterations")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    if not filename is None:
        plt.savefig(filename)
    np_vals = plot_to_numpy(fig)
    wandb.log({"train_test_loss": wandb.Image(np_vals)})
        
def combined_train_test_loss(combined_train_loss, combined_test_loss, filename=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # plot train on the left axis
    x_train = np.arange(0, len(combined_train_loss))
    recon, kl, triplet = list(zip(*combined_train_loss))
    ax[0].set_title("Train Loss Plot")
    ax[0].plot(x_train, recon, label="Reconstruction")
    ax[0].plot(x_train, kl, label="KL Divergence")
    ax[0].plot(x_train, triplet, label="Triplet")
    ax[0].set_ylim(0)
    # plot test on the right axis
    x_test = np.arange(0, len(combined_test_loss))
    recon, kl, triplet = list(zip(*combined_test_loss))
    ax[1].set_title("Test Loss Plot")
    ax[1].plot(x_test, recon, label="Reconstruction")
    ax[1].plot(x_test, kl, label="KL Divergence")
    ax[1].plot(x_test, triplet, label="Triplet")
    ax[1].set_ylim(0)
    # axis labels/title
    plt.xlabel("Num Iterations")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    if not filename is None:
        plt.savefig(filename)

    np_vals = plot_to_numpy(fig)
    wandb.log({"combined_train_test_loss": wandb.Image(np_vals)})

def save_reconstruction_model_ablation(combined_test_losses, model_configs, pivot_keys, num_trials=1):
    # initialize the plot
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    ax.set_title("Final Reconstruction Error Ablation")
    ax.set_xlabel(pivot_keys[0])
    ax.set_ylabel("Final Reconstruction Error") 
    # iterate over config objects
    last_recon_map = []    
    label_names = []
    x_axis_values = []
    for config_index, config_object in enumerate(model_configs):
        last_recons = []
        losses = combined_test_losses[config_index]    
        for trial_loss in losses:
            # unpack the loss
            recon, kl, triplet = trial_loss[-1]
            # get last recon loss
            last_recons.append(recon)
        last_recon_map.append(last_recons)
        # generate label name from pivot keys and values
        pivot_values = [config_object[pivot_key] for pivot_key in pivot_keys]
        x_axis_values.append(pivot_values[0])
        label_name = "".join([pivot_keys[i]+"_"+str(pivot_values[i])+"_" for i in range(len(pivot_values))])
        label_names.append(label_name)
    ax.violinplot(last_recon_map)
    ax.xaxis.set_ticks(np.arange(len(x_axis_values))+1) #set the ticks to be a
    ax.xaxis.set_ticklabels(x_axis_values) # change the ticks' names to x
    # save plot with wandb
    wandb.log({"End Recon Ablation": wandb.Image(plot_to_numpy(fig))}) 

def save_train_test_loss(trainer):
    epochs = trainer.epochs
    train_loss = trainer.train_loss
    test_loss = trainer.test_loss
    combined_train_loss = trainer.combined_train_loss
    combined_test_loss = trainer.combined_test_loss
    normal_save_name = None #f"{source_dir}/logs/{trainer.save_dir_name}/train_test_loss_{epochs}.png"
    train_test_loss(train_loss, test_loss, normal_save_name)
    combined_save_name = None #f"{source_dir}/logs/{trainer.save_dir_name}/combined_train_test_loss_{epochs}.png"
    combined_train_test_loss(combined_train_loss, combined_test_loss, combined_save_name)

def plot_triplet_loss(trainer):
    epochs = trainer.epochs
    combined_train_loss = trainer.combined_train_loss
    combined_test_loss = trainer.combined_test_loss
    train_triplet = [losses[2] for losses in combined_train_loss]
    test_triplet = [losses[2] for losses in combined_test_loss]
    normal_save_name = None #f"{source_dir}/logs/{trainer.save_dir_name}/train_test_loss_{epochs}.png"
    train_test_loss(train_triplet, train_triplet, normal_save_name, plot_name="triplet_loss_plot")
    combined_save_name = None #f"{source_dir}/logs/{trainer.save_dir_name}/combined_train_test_loss_{epochs}.png"

def save_similarity_image_sampling(trainer):
    # do an image sampling for both similarity space and reconstruction space 
    dm = trainer.data_manager
    model = trainer.model
    model.similarity_mode = False
    model.eval()     
    similarity_dim = model.similarity_dim
    reconstructive_dim = model.reconstructive_dim
    # get embedding vectors
    embedding_vectors = []
    num_channels = dm.image_test[0].shape[0]
    for i in range(len(dm.image_test)):
        input_image = torch.Tensor(dm.image_test[i])
        mean, logvar, z, xhat = model.forward(input_image.cuda())
        embedding_vectors.append(z.cpu().detach().numpy())
    embedding_vectors = np.array(embedding_vectors)
    # similarity space
    similarity_vectors = embedding_vectors[:, 0:similarity_dim]
    if np.shape(similarity_vectors)[-1] > 2:
        similarity_vectors = TSNE(n_components=2).fit_transform(similarity_vectors)
    # make the plot 
    fig = image_sampling.plot_binned_tsne_grid(similarity_vectors, embedding_vectors, model, num_channels=num_channels, dpi=500, num_x_bins=10, num_y_bins=20, title="2D MNIST 1-Digit Image Samples Without Slant Triplet Loss")
    #plt.savefig(f"{source_dir}/logs/{trainer.save_dir_name}/BinnedImageGrid.png")
    np_vals = plot_to_numpy(fig)
    wandb.log({"similarity_image_sampling" : wandb.Image(np_vals)})
    # reconstructive space
    reconstructive_vectors = embedding_vectors[:, similarity_dim:]
    if np.shape(reconstructive_vectors)[-1] > 2:
        reconstructive_vectors = TSNE(n_components=2).fit_transform(reconstructive_vectors)
    # make the plot 
    fig = image_sampling.plot_binned_tsne_grid(reconstructive_vectors, embedding_vectors, model, num_channels=num_channels, dpi=500, num_x_bins=10, num_y_bins=20, title="2D MNIST 1-Digit Image Samples Without Slant Triplet Loss")
    #plt.savefig(f"{source_dir}/logs/{trainer.save_dir_name}/BinnedImageGrid.png")
    np_vals = plot_to_numpy(fig)
    wandb.log({"reconstructive_image_sampling" : wandb.Image(np_vals)})

def plot_dimension_variance(trainer):
    dm = trainer.data_manager
    model = trainer.model
    model.eval()
    variances = []
    for i in range(len(dm.image_test)):
        input_image = torch.Tensor(dm.image_test[i])
        mean, logvar, z, xhat = model.forward(input_image.cuda())
        logvar = logvar.detach().cpu().numpy()
        variances.append(np.exp(logvar))
    variances = np.array(variances)
    variances = np.mean(variances, axis=0).squeeze()
    latent_dim = np.shape(variances)[0]
    # plot a bar graph
    fig = plt.figure()
    plt.bar(np.arange(latent_dim), variances)
    np_vals = plot_to_numpy(fig)
    wandb.log({"dimension_variance" : wandb.Image(np_vals)})
  
def save_image_sampling(trainer):
    dm = trainer.data_manager
    model = trainer.model
    model.eval()
    embedding_vectors = []
    num_channels = dm.image_test[0].shape[0]
    for i in range(len(dm.image_test)):
        input_image = torch.Tensor(dm.image_test[i])
        mean, logvar, z, xhat = model.forward(input_image.cuda())
        embedding_vectors.append(z.cpu().detach().numpy())
    embedding_vectors = np.array(embedding_vectors)
    tsne_points = embedding_vectors
    # run tsne if necessary 
    if np.shape(embedding_vectors)[-1] > 2:
        tsne_points = TSNE(n_components=2).fit_transform(embedding_vectors)
    
    fig = image_sampling.plot_binned_tsne_grid(tsne_points, embedding_vectors, model, num_channels=num_channels, dpi=500, num_x_bins=10, num_y_bins=20, title="2D MNIST 1-Digit Image Samples Without Slant Triplet Loss")
    #plt.savefig(f"{source_dir}/logs/{trainer.save_dir_name}/BinnedImageGrid.png")
    np_vals = plot_to_numpy(fig)
    wandb.log({"image_sampling" : wandb.Image(np_vals)})

"""
    Plots the masks of a learned mask vae triplet loss
"""
def plot_masks_as_heatmap(loss_object):
    try:
        # take the mask parameters from the loss object
        weights = loss_object.masks.data.detach().cpu().numpy()
        # plot the weights
        fig, ax = plt.subplots()
        num_masks = np.shape(weights)[0]
        width = 0.25
        for mask_index in range(num_masks):
            ax.bar(np.arange(0, np.shape(weights)[1]) + width*mask_index, weights[mask_index], width, alpha=0.5, label=f"{mask_index}")
        # log the plot
        np_vals = plot_to_numpy(fig)
        wandb.log({"Masks as Heatmap" : wandb.Image(np_vals)})
    except:
        pass

def save_metric_image_sampling(trainer):
    dm = trainer.data_manager
    model = trainer.model
    model.eval()
    metric_embedding = []
    no_metric_embedding = []
    num_channels = dm.image_test[0].shape[0]
    # no metric embedding
    model.use_metric = False
    for i in range(len(dm.image_test)):
        input_image = torch.Tensor(dm.image_test[i])
        mean, logvar = model.encode(input_image.cuda())
        mean = mean.squeeze()
        no_metric_embedding.append(mean.cpu().detach().numpy())
    # metric embedding
    model.use_metric = True
    for i in range(len(dm.image_test)):
        input_image = torch.Tensor(dm.image_test[i])
        mean, logvar = model.encode(input_image.cuda())
        mean = mean.squeeze()
        metric_embedding.append(mean.cpu().detach().numpy())
    # numpy 
    metric_embedding = np.array(metric_embedding)
    no_metric_embedding = np.array(no_metric_embedding)
    # run tsne if necessary 
    if np.shape(metric_embedding)[-1] > 2:
        metric_tsne_points = TSNE(n_components=2).fit_transform(metric_embedding)
    else:
        metric_tsne_points = metric_embedding
    if np.shape(no_metric_embedding)[-1] > 2:
        no_metric_tsne_points = TSNE(n_components=2).fit_transform(no_metric_embedding)
    else:
        no_metric_tsne_points = no_metric_embedding
    # metric figure 
    fig = image_sampling.plot_binned_tsne_grid(metric_tsne_points, metric_embedding, model, num_channels=num_channels, dpi=500, num_x_bins=12, num_y_bins=12, title="Metric Embedding Image Sampling")
    #plt.savefig(f"{source_dir}/logs/{trainer.save_dir_name}/BinnedImageGrid.png")
    np_vals = plot_to_numpy(fig)
    wandb.log({"Metric Embedding Image Sampling" : wandb.Image(np_vals)})
    # no metric figure
    fig = image_sampling.plot_binned_tsne_grid(no_metric_tsne_points, no_metric_embedding, model, num_channels=num_channels, dpi=500, num_x_bins=12, num_y_bins=12, title="No Metric Embedding Image Sampling")
    np_vals = plot_to_numpy(fig)
    wandb.log({"No Metric Embedding Image Sampling" : wandb.Image(np_vals)})

"""
    Plots the posterior means over time
"""
def save_localization_plots(trainer, localizers):
    if len(localizers) == 0:
        return
    model = trainer.model.to("cuda")
    model.similarity_mode = True
    model.eval()
    data_manager = trainer.data_manager
    num_localizers = len(localizers)
    num_queries = len(localizers[0].posterior_means)
    fig = plt.figure(figsize=(2*num_queries, 2*num_localizers))
    
    for i, localizer in enumerate(localizers):
        for j, mean in enumerate(localizer.posterior_means):
            ax = fig.add_subplot(num_localizers, num_queries + 1, i * (num_queries + 1) + j + 1)
            # decode mean
            mean = torch.Tensor(mean)
            decoded_mean = model.decode(mean.to("cuda")).cpu().detach().numpy().squeeze()
            ax.imshow(decoded_mean)
        # show reference image
        ax = fig.add_subplot(num_localizers, num_queries + 1, i * (num_queries + 1) + j + 2)
        reference_image = localizer.reference_data
        ax.imshow(reference_image)
    np_vals = plot_to_numpy(fig)
    wandb.log({"Localization Image" : wandb.Image(np_vals)})

"""
    Setup and run metadata localization testing
"""
def run_metadata_testing(trainer, num_queries, num_trials, k, noise_scale=0.0):
    if num_queries == 0 or num_trials == 0:
        return [], [0], [0]
    filename = f"{source_dir}/logs/{trainer.save_dir_name}/MetadataTesting.png"
    stan_file = f"{source_dir}/model.pkl"
    mode = trainer.localization
    # run metadata localization testing
    components = trainer.components
    noise = 0.0
    if not noise_scale is 0.0:
        noise = noise_scale
    else:
        noise = trainer.noise_scale
    localizers = morpho_mnist.metadata_localization.run_localizations(trainer, num_queries=num_queries, num_trials=num_trials, method=mode, components=components, stan_file=stan_file, k=k, noise_scale=noise)  
    data_manager = trainer.data_manager
    model = trainer.model.to("cuda")
    model.eval()
    # calculate losses
    metadata_losses = []
    localization_losses = []
    auc_losses = []
    for localizer in localizers:
        metadata_loss = morpho_mnist.metadata_localization.measure_localizer_metadata_loss(localizer, model, data_manager, components=components)     
        localization_loss = morpho_mnist.metadata_localization.measure_localizer_reference_loss(localizer)
        auc_loss = morpho_mnist.metadata_localization.measure_localizer_reference_auc(localizer, model, data_manager, components=components)
        metadata_losses.append(metadata_loss)
        localization_losses.append(localization_loss)
        auc_losses.append(auc_loss)

    metadata_variance = np.var(metadata_losses, axis=0)
    metadata_losses = np.mean(metadata_losses, axis=0)
    localization_variance = np.var(localization_losses, axis=0)
    localization_losses = np.mean(localization_losses, axis=0)
    auc_losses = np.mean(auc_losses)
    # make plots for these losses
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    metadata_train = np.arange(0, np.shape(metadata_losses)[0])
    ax[0].set_title("Metadata Loss Plot")
    ax[0].plot(metadata_train, metadata_losses, label="Metadata Loss")
    ax[0].fill_between(metadata_train, metadata_losses-metadata_variance, metadata_losses+metadata_variance, alpha=0.2)
    ax[0].set_ylim(0)
    # plot test on the right axis
    localization_train = np.arange(0, np.shape(localization_losses)[0])
    ax[1].set_title("Latent Loss Plot")
    ax[1].plot(localization_train, localization_losses, label="Latent Loss")
    ax[1].fill_between(localization_train, localization_losses-localization_variance, localization_losses+localization_variance, alpha=0.2)
    ax[1].set_ylim(0)
    # axis labels/title
    plt.xlabel("Num Queries")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    # save the plots
    if not filename is None:
        plt.savefig(filename)

    wandb.Image(plt) 
    return localizers, metadata_losses, auc_losses

"""
    Takes in a model and a triplet dataset and computes the "triplet importance"
    vmax = np.sum(global_mask_vector)
    vmax = np.sum(global_mask_vector)
    statistic about each of the latent units. This represents how important each of
    the latent units are to satisfying triplets. 
"""
def plot_triplet_importance(model, triplet_dataset):
    # calculate feature importances using the model and dataset
    data = compute_triplet_importance(model, triplet_dataset, loss_iters=1)
    # plot the data as a bar plot
    latent_dim = model.z_dim
    fig, ax = plt.subplots(1, 1, figsize=(0.3*latent_dim, 5))
    plt.title("Triplet Feature Importance Plot")
    latent_dims = np.arange(0, latent_dim)
    ax.bar(latent_dims, data)
    
    y_range = np.nanmax(np.abs(data)[np.abs(data) != np.inf])*1.2
    ax.set_ylim(-1*y_range, y_range)
    plt.xlabel("Latent Dimension")
    plt.ylabel("Triplet Importance")
    # set wider layout
    # save the plot
    np_vals = plot_to_numpy(fig)
    wandb.log({"Triplet Importance Statistics": wandb.Image(np_vals)})

    return fig

def save_plots_and_logs(trainer, config):
    save_train_test_loss(trainer)
    save_image_sampling(trainer)
    localizers, metadata_losses, auc_losses = run_metadata_testing(trainer, config["num_queries"], config["num_trials"], config["k"])
    
    save_localization_plots(trainer, localizers)
    
    return metadata_losses, auc_losses

