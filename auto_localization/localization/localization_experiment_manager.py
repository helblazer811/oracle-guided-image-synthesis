from tqdm import tqdm
import traceback
import wandb
import sys
sys.path.append("..")
import pickle
import numpy as np
import pandas as pd
import auto_localization.plotting.localization as localization_plotting
import auto_localization.localization.localizers.factory as localizer_factory
from auto_localization.localization.automatic_rollout import AutomaticRollout
from auto_localization.localization.indexed_automatic_rollout import IndexedAutomaticRollout
from auto_localization.localization.noise_model_selector import NoiseModelSelector

"""
    This class handles running specific localization rollouts 
    and modifying different parameters  
"""
class LocalizationExperimentManager():

    def __init__(self, localization_config=None, localization_oracle=None, metadata_oracle=None, data_manager=None, model=None, num_samples=None, model_type=None):
        self.localization_config = localization_config
        self.metadata_oracle = metadata_oracle
        self.localization_oracle = localization_oracle
        self.data_manager = data_manager
        self.model = model
        self.model_type = model_type
        if not self.model is None:
            self.latent_dim = self.model.z_dim
        self.num_samples = num_samples
        self.localizers = []
        self.localization_metrics = []
        self.pivot_keys = []
        self.config_indices = []
        self.trainer_name = "triplet"

    """
        Selects localization configuratino based on maximimum likelihood
    """
    def run_maximum_likelihood_selection(self):
        selector = NoiseModelSelector(
            model=self.model, 
            triplet_dataset=self.data_manager.triplet_test, 
            num_triplets=2000, 
            num_settings=200, 
            selection_parameters=["normalization", "k"],
            localizer_type=self.localization_config["localizer_type"],
        )
        # evaluate the triplets
        triplets = selector.evaluate_triplets()
        # evaluate the logistic response model
        parameters = selector.perform_brute_force(triplets)
        # generate configs
        configs = self.generate_configs()
        for config in configs:
            for param_key in parameters:
                config[param_key] = parameters[param_key]
        return configs

    """
        Generates specific localizer configs using the parameter 
        spaces given
    """
    def generate_configs(self):
        # get number of values with lists of values to explore
        for key in self.localization_config:
            value = self.localization_config[key]
            if isinstance(value, list):
                self.pivot_keys.append(key)
                
        # recursively generates list of new configs
        def recursive_helper(config_spec):
            configs = []
            # go through and make an empty object with each of the keys in the config
            pivot_key = None
            for key in config_spec:
                value = config_spec[key]
                if isinstance(value, list):
                    pivot_key = key
                    break
            # do recursive splitting
            if not pivot_key is None:
                for value in config_spec[pivot_key]:
                    new_config_spec = config_spec.copy()
                    new_config_spec[pivot_key] = value
                    new_configs = recursive_helper(new_config_spec)
                    configs.extend(new_configs)
            else:
                return [config_spec]
            return configs

        configs = recursive_helper(self.localization_config)
        return configs

    """
        Averages localizer metrics across trials
    """
    def get_average_localization_metrics(self, by_config=False):
        if not by_config:
            average_dict = {}
            variance_dict = {}
            for key in self.localization_metrics[0].keys():
                average_dict[key] = []
            # go through each metric dict
            for metric_dict in self.localization_metrics:
                for key in metric_dict.keys():
                    average_dict[key].append(np.array(metric_dict[key]))
            # average all keys
            for key in self.localization_metrics[0].keys():
                variance_dict[key] = np.var(average_dict[key], axis=0)
                average_dict[key] = np.mean(average_dict[key], axis=0)
            # average localizaiton metrics across trials
            return average_dict, variance_dict
        else:
            configs = []
            metrics_map = {}
            for index, config_index in enumerate(self.config_indices):
                config = self.localizer_configs[config_index]
                metrics = self.localization_metrics[index]
                # get all the pivot key values
                pivot_key_vals = "_".join([str(config[pivot_key]) for pivot_key in self.pivot_keys])
                if pivot_key_vals in metrics_map:
                    metrics_map[pivot_key_vals].append(metrics)
                else:
                    configs.append(config)
                    metrics_map[pivot_key_vals] = [metrics]

            averages = []
            variances = []
            for pivot_key_vals, all_metrics in metrics_map.items():
                # compute the averages of the given metrics
                average_dict = {}
                variance_dict = {}
                for key in all_metrics[0].keys():
                    average_dict[key] = [] 
                # go through each metric dict
                for metric_dict in all_metrics:
                    for key in metric_dict.keys():
                        average_dict[key].append(np.array(metric_dict[key]))
                # average all keys
                for key in all_metrics[0].keys():
                    variance_dict[key] = np.var(average_dict[key], axis=0)
                    average_dict[key] = np.mean(average_dict[key], axis=0)
                # average localizaiton metrics across trials
                averages.append(average_dict)
                variances.append(variance_dict)

            return configs, averages, variances

    def run(self):
        if "maximum_likelihood_selection" in self.localization_config and self.localization_config["maximum_likelihood_selection"]:
            self.localizer_configs = self.run_maximum_likelihood_selection()
        else:
            self.localizer_configs = self.generate_configs()
        # iterate through each of the generated configs
        trials = self.localization_config["trials"]
        indexed = self.localization_config["indexed"]
        similarity_mode = False
        if "similarity_mode" in self.localization_config:
            similarity_mode = self.localization_config["similarity_mode"]
        for trial in tqdm(range(trials)):
            # select a reference image 
            random_image_index = np.random.randint(0 , len(self.data_manager.image_test))
            reference_image = self.data_manager.image_test[random_image_index]
            reference_metadata = self.data_manager.metadata_test[random_image_index]
            trial_localizers = []
            trial_metrics = []
            for config_num, config in enumerate(self.localizer_configs):
                if "use_metric" in config:
                    use_metric = config["use_metric"]
                else:
                    use_metric = False
                # make a localizer
                localizer = localizer_factory.get_localizer_from_config(config)
                if similarity_mode:
                    self.model.similarity_mode = True
                if "num_pairs" in config:
                    num_pairs = config["num_pairs"]
                else:
                    num_pairs = 100
 
                localizer.initialize(
                    gen_model=self.model,
                    data=self.data_manager,
                    num_pairs=num_pairs,
                    ndim=config["latent_dim"],
                    lambda_pen_MCMV=config["lambda_pen_MCMV"] if "lambda_pen_MCMV" in config else 1.0,
                    lambda_latent_variance=config["lambda_latent_varinace"]  if "lambda_latent_variance" in config else 0.0,
                    k=config["k"],
                    k_relaxation=config["k_relaxation"] if "k_relaxation" in config else 1.0,
                    indexed=config["indexed"],
                )

                if not indexed:
                    # turn the metric on or off
                    self.model.use_metric = use_metric
                    # setup a localization experiment 
                    rollout = AutomaticRollout(
                        queries = config["num_queries"],
                        oracle = self.localization_oracle,
                        metadata_oracle = self.metadata_oracle,
                        image_dataset = self.data_manager.image_test,
                        metadata_dataset = self.data_manager.metadata_test,
                        localizer = localizer,
                        model = self.model,
                        reference_image = reference_image, 
                        reference_metadata = reference_metadata
                    )
                else:
                    # setup a localization experiment 
                    rollout = IndexedAutomaticRollout(
                        queries = config["num_queries"], 
                        oracle = self.localization_oracle,
                        metadata_oracle = self.metadata_oracle, 
                        image_dataset = self.data_manager.image_test,
                        localizer = localizer,
                        model = self.model, 
                        reference_index = random_image_index, 
                        reference_metadata = reference_metadata
                    )

                # run rollout
                rollout.run_localization(batched=config["batched"])
                # add to list
                self.localizers.append(localizer)
                # save localization metrics
                if not "record_metrics" in config or config["record_metrics"]:
                    rollout_metrics = rollout.get_localization_metrics()
                    #rollout.log_rollout(number=trial)
                    self.localization_metrics.append(rollout_metrics)
                self.config_indices.append(config_num)


    """
        Serializes localizers at given path
    """
    def save_localizers(self, dir_path, num=None):
        with open(dir_path+"/localizers.pkl", "wb") as f:
            for localizer in self.localizers:
                localizer.gen_model = None
            pickle.dump(self.localizers, f)

        with open(dir_path+"/localizer_metrics.pkl", "wb") as f:
            pickle.dump(self.localization_metrics, f)

    """
        Inverse of save_localizers
    """
    def load_localizers(self, dir_path):
        with open(dir_path+"/localizers.pkl", "rb") as f:
            self.localizers = pickle.load(f)

        with open(dir_path+"/localizer_metrics.pkl", "rb") as f:
            self.localization_metrics = pickle.load(f)

    def log_localization_metadata_distance(self):
        average_dict, variance_dict = self.get_average_localization_metrics(by_config=False)
        metadata_error = average_dict["metadata_loss"]
        # Log the final metadata error
        for error_val in metadata_error:
            wandb.log({"average_localization_error": error_val})

    """
        Logs the average of a set of localization performance metrics 
        with a bunch of k values
    """
    def log_performance_per_k(self):
        # Log performance of this model with the k constant
        configs, averages, variances = self.get_average_localization_metrics(by_config=True)
        for index, config in enumerate(configs):
            avgs = averages[index]
            log_dictionary = {}
            for metric in avgs.keys():
                print(metric)
                print(avgs[metric])
                if isinstance(avgs[metric], list) or isinstance(avgs[metric], np.ndarray):
                    log_dictionary["final_"+metric] = avgs[metric][-1]
            log_dictionary["k"] = config["k"]
            wandb.log(log_dictionary)

    """
        Generates localization plots 
    """
    def save_plots(self):
        try:
            self.log_performance_per_k()
        except Exception as e:
            print(traceback.format_exc())
        # plot localization plots 
        try:
            if len(self.pivot_keys) > 0:
                configs, averages, variances = self.get_average_localization_metrics(by_config=True)
                metrics = list(zip(averages, variances))
                localization_plotting.save_localization_ablation(metrics, configs, self.pivot_keys, num_trials=self.localization_config["trials"])
        except Exception as e:
            print(traceback.format_exc())
        try:
            localization_plotting.save_localization_metadata_distance_plots(self.get_average_localization_metrics())
        except Exception as e:
            print(traceback.format_exc())
            print(e)
        try:
            localization_plotting.save_localization_broken_down_metadata_distance_plots(self.get_average_localization_metrics(), self.metadata_oracle.component_weighting)
        except Exception as e:
            print(traceback.format_exc())
        try:
            localization_plotting.save_reference_percentile_plots(self.localization_metrics, self.get_average_localization_metrics())
        except Exception as e:
            print(traceback.format_exc())
        try:
            localization_plotting.save_nearest_neighbor_percentile_plots(self.localization_metrics, self.get_average_localization_metrics())
        except Exception as e:
            print(traceback.format_exc())
        try:
            plot_all = False
            if plot_all:
                for i in range(len(self.localizers)):
                    localization_plotting.save_embedding_with_posterior_path(self.model, self.localizers[i], self.data_manager.image_test)
            else:
                localization_plotting.save_embedding_with_posterior_path(self.model, self.localizers[0], self.data_manager.image_test)
        except Exception as e:
            print(traceback.format_exc())
        if hasattr(self.model, "similarity_mode"):
            self.model.similarity_mode = True
        try:
            localization_plotting.save_posterior_mean_over_time(self.model, self.data_manager, self.localizers)
        except Exception as e:
            print(e)
        try:
            localization_plotting.save_nearest_neighbor_image_over_time(self.model, self.data_manager, self.localizers)
        except Exception as e:
            traceback.print_exc()
            print(e)
        # takes a very long time
        #for loc_num, localizer in enumerate(self.localizers):
        try:
            plot_all_localizations = True
            if plot_all_localizations:
                for loc_num, localizer in enumerate(self.localizers):
                    localization_plotting.plot_posterior_mean_images(self.model, localizer, self.data_manager, plot_num=loc_num)
            else:
                localization_plotting.plot_posterior_mean_images(self.model, self.localizers[0], self.data_manager, plot_num=0)
        except Exception as e:
            print(e)
        # plot ablations 
        # TODO fix

