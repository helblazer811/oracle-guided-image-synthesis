import wandb
import sys
import torch
import traceback
import os
import pickle
import numpy as np
sys.path.append("../../")
import auto_localization.models.model_factory as model_factory
from auto_localization.training.trainer import TripletTrainer
from auto_localization.training.cycle_consistency_trainer import CycleConsistencyTrainer
import auto_localization.plotting.plotting as plotting
import auto_localization.plotting.metric_plotting as metric_plotting
import auto_localization.plotting.image_sampling as image_sampling
import auto_localization.plotting.localization as localization_plotting
import auto_localization.plotting.disentanglement as disentanglement_plotting
import auto_localization.plotting.response_model_plotting as response_model_plotting
from auto_localization.localization.automatic_rollout import AutomaticRollout
from auto_localization.localization.localization_experiment_manager import LocalizationExperimentManager
import auto_localization.localization.localizers.factory as localizer_factory
from tqdm import tqdm

save_path = os.path.dirname(os.path.dirname(__file__))+"/logs"

"""
    This is a wrapper class that handles training a model 
    with a given parameter configuration, tests localizations on it, 
    logs the loss and other metrics, and generates plots. 
"""
class BasicExperiment():

    def __init__(self, trials=1, data_manager=None, localization_oracle=None, metadata_oracle=None, experiment_config=None, model=None, run_plotting=True, do_setup=True, plot_regressions=False):
        if experiment_config is None:
            raise NotImplementedError
        self.device = "cuda"
        self.trails = trials
        self.experiment_config = experiment_config
        self.data_manager = data_manager
        self.localization_oracle = localization_oracle
        self.metadata_oracle = metadata_oracle
        self.plot_regressions = plot_regressions
        # uninitialized fields
        self.trainer = None
        self.model_config = None
        self.model_type = None
        self.model_predefined = not model is None
        self.model = model
        self.run_plotting = run_plotting
        self.do_setup = do_setup
        # run generic setup code
        if self.do_setup:
            self._setup()

    """
        Encapsulates the setup code
    """
    def _setup(self):
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        print("Setting Up Logging")
        #if self.do_setup:
        self._start_run()
        self._setup_save_directory()
        print("Setup Model Config")
        self._setup_model_config()
        print("Setting Up Model")
        if not self.model_predefined:
            self._setup_model()
        print("Setting Up Trianer")
        self._setup_trainer()
    
    def _setup_save_directory(self):
        # make save dir
        run_name = wandb.run.name
        save_dir_name = save_path+"/"+run_name
        os.mkdir(save_dir_name)

    def _setup_model_config(self):
        # set latent dim as sum of similarity dim and reconstruction dim if not there
        if not "latent_dim" in self.experiment_config["model_config"]:
            latent_dim = self.experiment_config["model_config"]["similarity_dim"] + self.experiment_config["model_config"]["reconstructive_dim"]
            self.experiment_config["model_config"]["latent_dim"] = latent_dim

    """
        Runs init, but using an input dictionary
    """
    @classmethod
    def from_config(cls, config, data_manager=None):
        return None  

    """
        Sets up a system of logging
    """
    def _start_run(self):
        self.run_object = wandb.init(
            project="latent-space-localization-final",
            entity="helblazer811",
            group=self.experiment_config["group"],
            config=self.experiment_config
        )
    
    """
        Initializes the model
    """
    def _setup_model(self):
        self.model_config = self.experiment_config["model_config"]
        self.model_type = self.model_config["model_type"]
        has_pretrained_run = "pretrained_run" in self.model_config
        if has_pretrained_run:
            run_name = self.model_config["pretrained_run"]
            save_path = os.path.dirname(os.path.dirname(__file__))+"/logs"
            directory_name = save_path+"/"+run_name
            # load model    
            model_path = directory_name + "/best_model.pkl"
            if not os.path.exists(model_path):
                raise Exception(f"Unrecognized pretrained path {run_name}")
            self.model = model_factory.get_model_from_config(self.model_type, self.model_config)
            try:
                self.model.load_state_dict(torch.load(model_path))
            except Exception as e:
                traceback.print_exc()
                raise Exception(f"Model configuration conflicts with pretrained path: {run_name}")
        elif not self.model_predefined:
            # Initialize the model
            self.model = model_factory.get_model_from_config(self.model_type, self.model_config)
        self.model.to(self.device)
        # print model architecture
        print(self.model)
        
    """
        Setting up the model Trainer
    """
    def _setup_trainer(self):
        trainer_name = self.experiment_config["trainer"] 
        if trainer_name == "triplet":
            self.trainer = TripletTrainer.from_config(
                self.experiment_config, 
                data_manager = self.data_manager,
                model = self.model
            )
        elif trainer_name == "cycle_consistency":
            self.trainer = CycleConsistencyTrainer.from_config(
                self.experiment_config, 
                data_manager = self.data_manager,
                model = self.model
            )
        else:
            raise Exception("Trainer not recognized : {}".format(self.trainer_name))

        self.trainer.epochs = self.experiment_config["epochs"]

    """
        Handles training of the model
    """
    def _train_model(self):
        # run the training
        self.trainer.train(epochs=self.experiment_config["epochs"])
        self.model.eval()

    """
        Runs localization tests on the trained model
    """
    def _run_localization(self):
        # set appropriate similarity mode
        localization_config = self.experiment_config["localization_config"]
        print(localization_config)
        self.localization_experiment_manager = LocalizationExperimentManager(
            localization_config,
            self.localization_oracle,
            self.metadata_oracle,
            self.data_manager,
            self.model,
            model_type=self.experiment_config["model_config"]["model_type"]
        )
        self.localization_experiment_manager.trainer_name = self.experiment_config["trainer"]
        # run the experiment
        self.localization_experiment_manager.run()

    """
        Generates plots based on the training of the model
        and the tested localization.
    """
    def _generate_plots(self):
        self.model.eval()
        try:
            plotting.plot_latent_covariance_triplet_satisfied_relationship(self.model, self.data_manager.triplet_test)
        except:
            print(traceback.format_exc())
        # plot response model probabilities
        try:
            response_model_plotting.plot_response_model_probabilities(
                self.model,
                self.data_manager.triplet_test, 
                maximum_likelihood_selection=self.experiment_config["localization_config"]["maximum_likelihood_selection"],
                default_k=self.experiment_config["localization_config"]["k"],
            )
        except:
            print(traceback.format_exc())
        try:
            response_model_plotting.average_embedding_magnitude(self.model, self.data_manager.image_test)
        except:
            print(traceback.format_exc())
        # plot reconstructive sampling
        image_sampling.plot_reconstructive_sampling(self.model, self.data_manager.image_test[list(range(0, 20))].squeeze().unsqueeze(1))
        # plot similarity reconstructive sampling
        image_sampling.plot_similarity_reconstructive_sampling(self.model, self.data_manager.image_test[list(range(0, 20))].squeeze().unsqueeze(1))
        # plot localization plots
        self.localization_experiment_manager.save_plots()
        # plot the masks
        if hasattr(self.model, "loss_function"):
            plotting.plot_masks_as_heatmap(self.model.loss_function)
        # Plot 
        try:
            component_weighting = self.experiment_config["dataset_config"]["component_weighting"]
            for feature_index in range(0, len(self.data_manager.triplet_train.metadata_dataset[0])):
                if not component_weighting[feature_index] > 0.0:
                   continue 
                disentanglement_plotting.plot_feature_embedding(self.model, self.data_manager, feature_index=feature_index)
        except:
            pass
        # plot triplet importance of the model latent units
        try:
            plotting.plot_dimension_variance(self.trainer)
        except:
            pass
        #plotting.plot_triplet_importance(self.model, self.data_manager.triplet_test)
        if not self.model_predefined:
            # plot various training information
            try:
                plotting.save_train_test_loss(self.trainer)
            except:
                pass
            # plot triplet loss
            try:
                plotting.plot_triplet_loss(self.trainer)
            except:
                pass
        # plot metric learning plots
        if self.experiment_config["trainer"] == "metric":
            metric_plotting.plot_mahalanobis_vs_identity(self.trainer)
            plotting.save_metric_image_sampling(self.trainer)     
        else:
            if "similarity_mode" in self.experiment_config["localization_config"] and False:
                if self.experiment_config["localization_config"]["similarity_mode"]:
                    plotting.save_similarity_image_sampling(self.trainer)     
                self.trainer.model.similarity_mode = self.experiment_config["localization_config"]["similarity_mode"] 
        # plot disentanglement plots
        try:
            component_weighting = self.experiment_config["dataset_config"]["component_weighting"]
            if self.plot_regressions:
                disentanglement_plotting.plot_isotonic_regressions(self.model, self.data_manager, component_weighting)        
        except:
            pass
        # plot global mask heatmap if loss function is "GlobalMaskVAETripletLoss"
        if self.experiment_config["model_config"]["loss_name"] == "GlobalMaskVAETripletLoss":
            # get global mask vector
            print("plotting global mask heatmap")
            global_mask_vector = self.model.loss_function.get_mask_activation_vector()
            disentanglement_plotting.plot_global_mask_heatmap(global_mask_vector)

    """
        Saves the model and serializes the localization data in a 
        directory based on the current wandb run name. 
    """
    def _save_data(self):
        run_name = wandb.run.name
        save_dir_name = save_path+"/"+run_name
        # save run params
        params_path = save_dir_name + "/params.pkl"
        with open(params_path, "wb") as f:
            pickle.dump(self.experiment_config, f)
        # save localizers
        self.localization_experiment_manager.save_localizers(save_dir_name)
        # save combined training loss
        self.trainer.save_combined_loss(save_dir_name)
    
    """
        Does the inverse of _save_data by loading the files into 
        this object
    """
    def load_data(self, directory_name, best_model=False, overwrite_config=False):
        # load params
        if not overwrite_config:
            params_path = directory_name + "/params.pkl"
            with open(params_path, "rb") as f:
                self.experiment_config = pickle.load(f)
        # setup
        self._setup()    
        # load model    
        if best_model:
            model_path = directory_name + "/best_model.pkl"
            self.model.load_state_dict(torch.load(model_path), strict=False)
        else:
            model_path = directory_name + "/model.pkl"
            self.model.load_state_dict(torch.load(model_path), strict=False)
        self.model.eval()
        # load localizers        
        self.localization_experiment_manager = LocalizationExperimentManager(self.experiment_config["localization_config"], data_manager=self.data_manager, model=self.model) 
        try:
            self.localization_experiment_manager.load_localizers(directory_name)
        except:
            print("Failed to load localizers")
        # load combined loss
        self.trainer.load_combined_loss(directory_name)
        self.trainer.data_manager = self.data_manager

    def load_best_model(self):
        run_name = wandb.run.name
        save_path = os.path.dirname(os.path.dirname(__file__))+"/logs"
        directory_name = save_path+"/"+run_name
        # load model    
        model_path = directory_name + "/best_model.pkl"
        if not os.path.exists(model_path):
            return
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self):
        run_name = wandb.run.name
        save_dir_name = save_path+"/"+run_name
        # serialize model
        model_path = save_dir_name + "/model.pkl"
        torch.save(self.model.state_dict(), model_path)

    def save_run_params(self):
        run_name = wandb.run.name
        save_dir_name = save_path+"/"+run_name
        # save run params
        params_path = save_dir_name + "/params.pkl"
        with open(params_path, "wb") as f:
            pickle.dump(self.experiment_config, f)

    def run(self):
        # TODO setup the ability to do multiple trails
        self.save_run_params()
        if not self.model_predefined:
            print("Training Model")
            self._train_model()
        self.save_model()
        # load best model
        self.load_best_model()
        self.model.eval()
        print("Running Localization")
        self._run_localization()
        print("Saving data")
        self._save_data()
        if self.run_plotting:
            print("Generating Plots")
            self._generate_plots()
        self.run_object.finish()

