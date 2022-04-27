import os
import json
import glob

reserved = ["attributes", "component_weighting"]

"""
    Save the given config objects in a certain directory
"""
def save_configs(model_configs, experiment_config, save_directory, split_trials=True):
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    trials = experiment_config["trials"]
    # get number of unique json files
    num_unique_json = len(glob.glob(f"{save_directory}/*.json"))
    config_files = []
    # save it as a json object in the given directory
    for index in range(len(model_configs)):
        config = experiment_config.copy()
        config["model_config"] = model_configs[index]
        if split_trials:
            config["trials"] = 1
            for trial_index in range(trials):
                out_index = num_unique_json + index * trials + trial_index
                save_path = f'{save_directory}/config{out_index}.json'
                with open(save_path, 'w') as fp:
                    json.dump(config, fp)
                config_files.append(save_path)
        else:
            save_path = f'{save_directory}/config{num_unique_json+index}.json'
            with open(save_path, 'w') as fp:
                json.dump(config, fp)
            config_files.append(save_path)
    return config_files

"""
    Generate a list of configs for the models from 
    experiment config. 
"""
def generate_experiment_split(experiment_config, model=True):
    model_configs = []
    pivot_keys = []
    # get number of values with lists of values to explore
    if model:
        config = experiment_config["model_config"]
    else:
        config = experiment_config["localization_config"]

    for key in config:
        value = config[key]
        if isinstance(value, list):
            pivot_keys.append(key)
            
    # recursively generates list of new configs
    def recursive_helper(config_spec):
        configs = []
        # go through and make an empty object with each of the keys in the config
        pivot_key = None
        for key in config_spec:
            value = config_spec[key]
            if isinstance(value, list) and not key in reserved:
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
        
    configs = recursive_helper(config)
    
    return configs, pivot_keys


