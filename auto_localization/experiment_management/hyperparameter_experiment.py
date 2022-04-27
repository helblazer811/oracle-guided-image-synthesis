import wandb
import sys
import torch
import numpy as np
import os
import pickle
sys.path.append("../../")
import auto_localization.models.model_factory as model_factory
from auto_localization.training.trainer import TripletTrainer
from auto_localization.experiment_management.basic_experiment import BasicExperiment
import auto_localization.plotting.plotting as plotting
import auto_localization.plotting.metric_plotting as metric_plotting
import auto_localization.plotting.localization as localization_plotting
from auto_localization.localization.automatic_rollout import AutomaticRollout
from auto_localization.localization.localization_experiment_manager import LocalizationExperimentManager
import auto_localization.localization.localizers.factory as localizer_factory
import auto_localization.experiment_management.util as util
from tqdm import tqdm
import json

save_path = os.path.dirname(os.path.dirname(__file__))+"/logs"

"""
    This is a wrapper class that handles training a model 
    with a given parameter configuration, tests localizations on it, 
    logs the loss and other metrics, and generates plots. 
"""
class HyperparameterExperiment():

    def __init__(self, trials=1, data_manager=None, localization_oracle=None, metadata_oracle=None, experiment_config=None, model=None, run_plotting=True):
        if experiment_config is None:
            raise NotImplementedError
        self.device = "cuda"
        self.trails = trials
        self.experiment_config = experiment_config
        self.trials = self.experiment_config["trials"]
        self.data_manager = data_manager
        self.localization_oracle = localization_oracle
        self.metadata_oracle = metadata_oracle
        self.model_configs = []
        self.pivot_keys = []
        self.metrics_map = []
        self.combined_test_losses = []
        self.run_plotting = run_plotting
        # run generic setup code
        print("Setting Up Model Configs")
        self.log_path = os.path.expanduser("~")+"/p-crozell3-0/latent-space-localization/auto_localization/logs/"
        self.model_configs, self.pivot_keys = util.generate_experiment_split(self.experiment_config, model=True)

    """
        Takes in a BasicExperiment run's localizer metrics and finds the 
        cross trial average and variance
    """
    def _calculate_summary_localizer_metrics(self, localization_metrics):
        average_dict = {}
        variance_dict = {}
        for key in localization_metrics[0][0].keys():
            average_dict[key] = [] 
            variance_dict[key] = []
        # go through each metric dict
        for metric_dict_average, metric_dict_variance in localization_metrics:
            for key in metric_dict_average.keys():
                average_dict[key].append(np.array(metric_dict_average[key]))
                variance_dict[key].append(np.array(metric_dict_variance[key]))
        # average all keys
        for key in localization_metrics[0][0].keys():
            variance_dict[key] = np.mean(variance_dict[key], axis=0)
            average_dict[key] = np.mean(average_dict[key], axis=0)
        # average localizaiton metrics across trials
        return average_dict, variance_dict

    """
        Generates plots based on the training of the model
        and the tested localization.
    """
    def generate_plots(self):
        # sample mean and sample variance for trial metrics
        model_config_localizer_metrics = []
        for config_index, config in enumerate(self.model_configs):
            # mean across different trials of the same model
            config_metrics = self.metrics_map[config_index]
            average_dict, variance_dict = self._calculate_summary_localizer_metrics(config_metrics)
            model_config_localizer_metrics.append((average_dict, variance_dict))
        # plot ablation plot based on the pivot features
        # make a function that takes pivot keys, configs, and metrics
        localization_plotting.save_model_ablation(model_config_localizer_metrics, self.model_configs, self.pivot_keys, num_trials=self.trials) 
        # plot end localization error ablation
        localization_plotting.save_end_localization_model_ablation(self.metrics_map, self.model_configs, self.pivot_keys, num_trials=self.trials) 
        # plot an ablation feature on the x axis and the reconstruction error on the Y
        plotting.save_reconstruction_model_ablation(self.combined_test_losses, self.model_configs, self.pivot_keys, num_trials=self.trials)

    """
        Saves the model and serializes the localization data in a 
        directory based on the current wandb run name. 
    """
    def _save_data(self):
        # make save dir
        run_name = wandb.run.name
        save_dir_name = save_path+"/"+run_name
        os.mkdir(save_dir_name)
        # save run params
        params_path = save_dir_name + "/params.pkl"
        with open(params_path, "wb") as f:
            pickle.dump(self.experiment_config, f)
        # save metrics
        metrics_path = save_dir_name + "/metrics.pkl"        
        with open(metrics_path, "wb") as f:
            pickle.dump(self.metrics_map, f)
    
    """
        Inverse of save_data, takes in a directory path where the data
        was saved and loads them up into this object
    """
    def load_data(self, group_directory_list):
        
        def is_same(dict_1, dict_2):
            for key in dict_1:
                if not key in dict_1 or not key in dict_2:
                    continue
                if not dict_1[key] == dict_2[key]:
                    return False
            return True

        # go through and deserialize the basic experiment object for each given 
        abs_path = ""
        self.metrics_map = []
        # load configs
        name_to_config = {}
        for directory_name in group_directory_list:
            # deserialize the params
            with open(self.log_path+directory_name+"/params.pkl", "rb") as f:
                config = pickle.load(f)
            name_to_config[directory_name] = config
         
        for model_config in self.model_configs:
            metrics = []
            combined_test_losses = []
            for directory_name in group_directory_list:
                if is_same(name_to_config[directory_name]["model_config"], model_config):
                    basic_experiment = BasicExperiment(experiment_config=model_config, do_setup=False)
                    basic_experiment.load_data(self.log_path+directory_name)
                    metric = basic_experiment.localization_experiment_manager.get_average_localization_metrics() 
                    metrics.append(metric)
                    combined_test_loss = basic_experiment.trainer.combined_test_loss
                    combined_test_losses.append(combined_test_loss)
            self.combined_test_losses.append(combined_test_losses)
            self.metrics_map.append(metrics)

    """
        Runs a set of experiments for a bunch of models
    """
    def _run_model_experiments(self):
        # go through each config
        # run a basic experiment
        for model_config in self.model_configs:
            metrics = []
            succesful_trials = 0
            while succesful_trials < self.trials:
                # error handling
                try:
                    current_config = self.experiment_config.copy()
                    current_config["model_config"] = model_config
                    basic_experiment = BasicExperiment(
                        data_manager=self.data_manager,
                        localization_oracle=self.localization_oracle,
                        metadata_oracle=self.metadata_oracle,
                        experiment_config=current_config
                    )
                    basic_experiment.run()
                    # save the metrics
                    trial_metrics = basic_experiment.localization_experiment_manager.get_average_localization_metrics()
                    metrics.append(trial_metrics)
                    succesful_trials += 1
                except Exception as e:
                    print(e)
            # save the metrics for this model config
            self.metrics_map.append(metrics)
    
    def finish_sweep(self):
        # label config as hyperparameter config
        self.experiment_config["hyperparameter"] = True
        # init a run
        this_run = wandb.init(project="latent-space-localization", group=self.experiment_config["group"], entity="helblazer811", config=self.experiment_config)
        # log plots with the run
        print("Plotting")
        self.generate_plots()
        print("Save files")
        self._save_data()
        # finish the run
        this_run.finish()

    """
        Saves each of the config files as json files in the given directory
        path
    """ 
    def save_configs(self, save_directory, split_trials=True):
        # save the experiment config
        with open(f'{save_directory}/experiment_config.json', 'w') as fp:
            json.dump(self.experiment_config, fp)
        util.save_configs(self.model_configs, self.experiment_config, save_directory)    

    def run(self):
        print("Train models")
        self._run_model_experiments()
        print("Finish Sweep")
        self.finish_sweep()
        
