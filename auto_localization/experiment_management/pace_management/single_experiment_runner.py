import sys
sys.path.append("../../../")
from auto_localization.experiment_management.basic_experiment import BasicExperiment
from auto_localization.dataset_management.data_manager import DataManager
from auto_localization.dataset_management.data_manager_factory import construct_data_manager
import numpy as np
import argparse
import json
import torch
torch.backends.cudnn.enabled = False

"""
    Loads config object from a json file at the given path
"""
def load_config(path):
    with open(path, "r") as f:
        config_dict = json.load(f)
        return config_dict

def run_experiment(config_path):
    experiment_config = load_config(config_path)
    dataset_config = experiment_config["dataset_config"]
    # make data manager
    data_manager, localization_oracle = construct_data_manager(dataset_config)
    # setup the experiment manager
    print("Setting up Experiment")
    experiment = BasicExperiment(
        data_manager=data_manager, 
        localization_oracle=localization_oracle, 
        metadata_oracle=localization_oracle, 
        experiment_config=experiment_config
    )
    # run the experiment
    print("Running Experiment")
    experiment.run()

if __name__ == "__main__":    
    # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    config_path = args.config_path
    run_experiment(config_path)
