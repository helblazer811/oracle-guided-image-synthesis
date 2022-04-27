import matplotlib.pyplot as plt
import wandb
import sys
sys.path.append("../../")
from auto_localization.plotting.util import *

""" 
    Plots the difference in triplet loss using a learned 
    Mahalanobis metric compared to without. 
"""
def plot_mahalanobis_vs_identity(trainer):
    # get the identity triplet loss
    identity_triplet_loss = trainer.identity_triplet_loss
    # get the regular triplet loss
    triplet_loss = [loss for (_, _, loss) in trainer.combined_test_loss]
    # plot them on an axis
    fig, ax = plt.subplots()
    identity_x = np.arange(0, len(identity_triplet_loss))
    ax.plot(identity_x, identity_triplet_loss, label="Identity Triplet Loss")
    regular_x = np.arange(0, len(triplet_loss))
    ax.plot(regular_x, triplet_loss, label="Metric Learning Triplet Loss")
    # axis labels/title
    plt.title("Metric Learning vs Identity Triplet Loss Plot")
    plt.xlabel("Num Epochs")
    plt.ylabel("Triplet Loss")
    plt.legend(loc="upper left")
    # save the axis 
    np_vals = plot_to_numpy(fig)
    wandb.log({"mahalanobis_vs_identity": wandb.Image(np_vals)})

