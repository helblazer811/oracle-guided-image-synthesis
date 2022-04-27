import sys
sys.path.append("../..")
from auto_localization.localization.noise_model_selector import NoiseModelSelector
from auto_localization.plotting.util import * 
import wandb
import matplotlib.pyplot as plt
import torch
import numpy as np

def average_embedding_magnitude(model, image_dataset):
    embedding_magnitudes = []
    num_samples = 200
    for i in range(num_samples):
        image = image_dataset[i].cuda()
        mean, _, _, _ = model.forward(image)
        embedding_magnitudes.append(np.linalg.norm(mean.detach().cpu().numpy()))

    average_embedding_magnitude = np.mean(embedding_magnitudes)

    wandb.log({"average_embedding_magnitude": average_embedding_magnitude})

def plot_response_model_probabilities(model, triplet_dataset, default_k=1.0, maximum_likelihood_selection=False):
    noise_model_selector = NoiseModelSelector(model, triplet_dataset=triplet_dataset, localizer_type="RandomLogistic")
    # evaluate the triplets
    triplets = noise_model_selector.evaluate_triplets()
    # evaluate the logistic response model
    if not maximum_likelihood_selection:
        best_setting = {"k": default_k, "normalization": 0}
    else:
        best_setting = noise_model_selector.perform_brute_force(triplets)
    logistic_probs = noise_model_selector.compute_success_probabilities(best_setting, triplets)
    logistic_probs = np.stack(logistic_probs).squeeze()
    # evaluate the triplet response model
    noise_model_selector = NoiseModelSelector(model, triplet_dataset=triplet_dataset, localizer_type="RandomTriplet")
    if not maximum_likelihood_selection:
        best_setting = {"k": default_k, "normalization": 0}
    else:
        best_setting = noise_model_selector.perform_brute_force(triplets)
    triplet_probs = noise_model_selector.compute_success_probabilities(best_setting, triplets)
    triplet_probs = np.stack(triplet_probs).squeeze()
    # make a violin plot of the evaluated response model probabilities
    fig, ax = plt.subplots(1, 1)
    ax.violinplot([triplet_probs, logistic_probs])
    k_val = best_setting["k"]
    ax.set_title(f"Response Model Probabilities (k = {k_val})")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Triplet', 'Logistic'])
    ax.set_ylabel("Probabilities")
    # log it
    np_vals = plot_to_numpy(fig)
    wandb.log({"Response Model Probabilities": wandb.Image(np_vals)})
