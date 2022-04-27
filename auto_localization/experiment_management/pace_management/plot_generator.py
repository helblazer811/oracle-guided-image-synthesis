import wandb
import sys
sys.path.append("../../../")
from auto_localization.experiment_management.hyperparameter_experiment import HyperparameterExperiment
import wandb
import argparse 
import json

"""
    Class that manages taking a group of run
    processes and generatign plots from them
"""
class PlotGenerator():

    def __init__(self, group_name, experiment_config):
        self.group_name = group_name
        self.experiment_config = experiment_config
        self.jobs_name_list = self._get_wandb_names_for_group()
        self.hyperparameter_experiment = self._make_dummy_hyperparameter_experiment()

    def _get_wandb_names_for_group(self):
        # connect with wandb and load all run names with the group group_name
        api = wandb.Api()        
        runs = api.runs(path="helblazer811/latent-space-localization", filters={"group": self.group_name, "State": "finished", })
        filtered_runs = []
        for run in runs:
            if not "hyperparameter" in run.config:
                filtered_runs.append(run)
        run_names = [run.name for run in filtered_runs]
        return run_names

    """
        Makes a dummy hyperparameter tuning experiment
        for running logging
    """
    def _make_dummy_hyperparameter_experiment(self):
        experiment = HyperparameterExperiment(experiment_config=self.experiment_config)
        experiment.load_data(self.jobs_name_list)
        return experiment
    
    def run_plotting(self):
        # runs plotting function   
        self.hyperparameter_experiment.finish_sweep()
       
if __name__ == "__main__":
    # take in a directory of configs and run a job for each config 
    parser = argparse.ArgumentParser()
    parser.add_argument("group_name", type=str)
    parser.add_argument("experiment_config", type=str)
    args = parser.parse_args()
    group_name = args.group_name
    experiment_config = args.experiment_config
    experiment_config = json.loads(experiment_config)
    # run plotter
    plotter = PlotGenerator(group_name, experiment_config)
    plotter.run_plotting()
