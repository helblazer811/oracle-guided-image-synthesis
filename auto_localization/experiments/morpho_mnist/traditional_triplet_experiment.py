import sys
sys.path.append("../../../")
from auto_localization.experiment_management.basic_experiment import BasicExperiment
from auto_localization.dataset_management.data_manager import DataManager
import auto_localization.dataset_management.data_manager_factory as data_manager_factory
from datasets.morpho_mnist.dataset import MetadataDataset, ImageDataset, TripletDataset
from auto_localization.oracles.metadata_oracle import MetadataOracle
from auto_localization.oracles.indexed_metadata_oracle import IndexedMetadataOracle
from auto_localization.oracles.indexed_class_oracle import IndexedClassOracle
from auto_localization.oracles.oracle import EnsembleOracle
from datasets.morpho_mnist.measure import measure_image
import torch
import numpy as np
import argparse

def run_experiment(experiment_config):
    dataset_config = experiment_config["dataset_config"]
    data_manager, localization_metadata_oracle = data_manager_factory.construct_morpho_mnist(dataset_config)

    basic_experiment = BasicExperiment(
        data_manager=data_manager,
        localization_oracle=localization_metadata_oracle,
        metadata_oracle=localization_metadata_oracle,
        experiment_config=experiment_config
    )
    # run the experiment
    print("Running Experiment")
    basic_experiment.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group_name", type=str)
    args = parser.parse_args()
    group_name = args.group_name

    experiment_config = {
        "trials": 1,
        "epochs": 50,
        "lr": 1e-3,
        "batch_size": 256,
        "trainer": "triplet",
        "weight_decay": 0.0,
        "gamma": 0.999,
        "optmizer_type": "adam",
        "lambda_warmup_epochs": 1, 
        "triplet_warm_start": 1,
        "recon_warm_start": 1,
        "recon_zero_start": 0,
        "mcmv_sampling": False,
        "group": group_name,
        "localization_config": {
            "indexed": False,
            "trials": 20,
            "k": 1.0,
            "localizer_type": "BayesianTriplet",
            "num_queries": 30,
            "similarity_mode": True,
            "similarity_dim": 6,
            "normalization": 0,
            "latent_dim": 6,
            "maximum_likelihood_selection": False,
        },
        "model_config": {
            "model_type": "IsolatedVAE",
            "latent_dim": 6,
            "similarity_dim": 6,
            "reconstructive_dim": 0,
            "in_shape": 32,
            "d": 32,
            "layer_count": 4,
            "channels": 1,
            "loss_name": "BayesianTripletLoss",
            "sub_loss_type": "isolated", # can be isolated, combined, or bayesian
            "masks": False,
            "adam_beta_1": 0.9,
            "adam_beta_2": 0.999,
            "kl_beta": 1e-2, 
            "recon_beta": 1.0,
            "triplet_beta": 1.0,
            "triplet_margin": 0.1,
            "uncertainty_constant": False, 
            "similarity_batchnorm": False,
            "triplet_mining": False,
            "l2_normalization": False,
            "attributes": [0, 1, 2, 3, 4, 5],
            "component_weighting": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # weights slant and thickness
            "masks": False,
            "isotropic_variance": False,
            "bce": True,
            "gradient_clipping": False,
            "indexed": True,
        },
        "dataset_config": {
            "component_weighting": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # weights slant and thickness
            "attribute_return": True,
            "which_digits": [1],
            "one_two_ratio": 0.0, 
            "batch_size": 256,
            "dataset_name": "MorphoMNIST",
            "indexed": True,
            "num_workers": 6,
            "single_feature_triplet": False,
            "inject_triplet_noise": 0.0
        }
    }
 
    run_experiment(experiment_config)
