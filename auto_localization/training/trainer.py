import sys
sys.path.append("../..")
from auto_localization.models.BasicVAE import BasicVAE
from auto_localization.models.IsolatedVAE import IsolatedVAE
from auto_localization.models.MaskedVAE import MaskedVAE
from auto_localization.models.MaskedVAEIsolated import MaskedVAEIsolated
from auto_localization.models.loss.beta_tcvae_loss import BetaTCVAELoss
from auto_localization.models.BetaTCVAE import BetaTCVAE
from auto_localization.models.LearnedMaskedVAE import LearnedMaskedVAE
from auto_localization.models.CelebABetaVAE import CelebABetaVAE
from auto_localization.models.loss.bayesian_triplet_loss import BayesianTripletLoss
from auto_localization.models.CelebAVAE import CelebAVAE
from auto_localization.training.triplet_mining_wrapper import TripletMiningDatasetWrapper 
from auto_localization.training.scheduler import CycleScheduler
from auto_localization.training.training_test import reconstruction_of_metadata
import auto_localization.training.training_test as training_test
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.cuda as cutorch
import numpy as np
#import automatic_testing
import os
import wandb
from ray.tune.integration.wandb import wandb_mixin
from ray import tune
import pickle
import traceback

def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = 0.0
    for dictionary in dict_list:
        for key in dictionary.keys():
            loss = dictionary[key].detach().cpu().item() 
            mean_dict[key] += loss
    for key in dict_list[0].keys():
        mean_dict[key] /= len(dict_list)

    return mean_dict

def add_dicts(dict_a, dict_b):
    if dict_a is None:
        return dict_b
    if dict_b is None:
        return dict_a
    sum_dict = {}
    for key in dict_a.keys():
        if isinstance(dict_a[key], torch.Tensor):
            dict_a[key] = dict_a[key].detach().cpu().item()
        if isinstance(dict_b[key], torch.Tensor):
            dict_b[key] = dict_b[key].detach().cpu().item()
        sum_dict[key] = dict_a[key] + dict_b[key] 

    return sum_dict

def divide_dict(dictionary, value):
    divided_dict = {}
    for key in dictionary.keys():
        divided_dict[key] = dictionary[key] / value
    
    return divided_dict 

def detach_loss_dict(loss_dict):
    output_dict = {}
    for key in loss_dict.keys():
        if isinstance(loss_dict[key], torch.Tensor):
            output_dict[key] = loss_dict[key].detach().cpu()

    return output_dict

class TripletTrainer():
    
    def __init__(self, model=None, data_manager=None, lr=0.001, batch_size=128, localization="MCMV", triplet_beta=1.0, warm_start=1, kl_max=-1, triplet_mining=True, mcmv_sampling=False, weight_decay=1.0, gamma=1.0, optimizer_type="adam", lambda_warmup_epochs=1, triplet_warm_start=1, kl_beta=1.0, recon_warm_start=1, recon_beta=1.0, recon_zero_start=1, gradient_clipping=False, clip_value=None, cycle_consistent_warmup=0, cycle_consistent_beta=0.0):
        use_cuda = True
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = model
        # hyperparameters
        self.kl_max = kl_max
        self.kl_beta = kl_beta
        self.batch_size = batch_size
        self.mcmv_sampling = mcmv_sampling
        self.optimizer_type = optimizer_type
        self.warm_start = warm_start 
        self.annealing_amount = self.kl_max / self.warm_start
        self.lr = lr
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.recon_warm_start = recon_warm_start
        self.localization = localization
        self.data_manager = data_manager
        self.recon_beta = recon_beta
        self.recon_zero_start = recon_zero_start
        self.triplet_beta = triplet_beta
        self.triplet_mining = triplet_mining
        self.triplet_warm_start = triplet_warm_start
        self.lambda_warmup_epochs = lambda_warmup_epochs
        self.gradient_clipping = gradient_clipping
        self.cycle_consistent_warmup = cycle_consistent_warmup
        self.cycle_consistent_beta = cycle_consistent_beta
        self.clip_value = clip_value
        self.reparam_lambda = 0.0
        # setup logging information storage
        self.train_loss = []
        self.combined_train_loss = [] 
        self.test_loss = []
        self.combined_test_loss = []
        # setup triplet mining
        if self.triplet_mining:
            self._setup_triplet_mining()
        #setup optimizers
        self._setup_optimizers()

    """
        Sets up a learning rate scheduler
    """
    def _setup_scheduler(self, epochs, scheduler_type="exponential"):
        if scheduler_type is "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma = self.gamma,
            )
        elif scheduler_type is "cycle":
            scheduler = CycleScheduler(
                self.optimizer,
                self.lr,
                n_iter=len(self.data_manager.image_train_loader) * epochs,
                momentum=None,
                warmup_proportion=0.05,
            )
            
        return scheduler

    """
        Creates TripletMNIST Trainer from config file
    """
    @classmethod
    def from_config(cls, config, data_manager=None, model=None):
        return cls(
            batch_size = config["dataset_config"]["batch_size"],
            lr = config["lr"],
            localization = config["localization_config"]["localizer_type"],
            triplet_beta = config["model_config"]["triplet_beta"] if "triplet_beta" in config["model_config"] else 0.0,
            kl_beta = config["model_config"]["kl_beta"] if "kl_beta" in config["model_config"] else 1.0,
            recon_beta = config["model_config"]["recon_beta"] if "recon_beta" in config["model_config"] else 1.0,
            kl_max = config["model_config"]["kl_max"] if "kl_max" in config["model_config"] else config["model_config"]["kl_beta"],
            warm_start = config["model_config"]["warm_start"] if "warm_start" in config["model_config"] else 1,
            lambda_warmup_epochs = config["lambda_warmup_epochs"] if "lambda_warmup_epochs" in config else 1, 
            triplet_mining = config["model_config"]["triplet_mining"] if "triplet_mining" in config["model_config"] else 0,
            mcmv_sampling = config["mcmv_sampling"] if "mcmv_sampling" in config else False,
            weight_decay = config["weight_decay"] if "weight_decay" in config else 0.0,
            triplet_warm_start = config["triplet_warm_start"] if "triplet_warm_start" in config else 1,
            recon_warm_start = config["recon_warm_start"] if "recon_warm_start" in config else 1,
            recon_zero_start = config["recon_zero_start"] if "recon_zero_start" in config else 1,
            gamma = config["gamma"] if "gamma" in config else 1.0,
            optimizer_type = config["optimizer_type"] if "optimizer_type" in config else "adam",
            clip_value = config["clip_value"] if "clip_value" in config else None,
            gradient_clipping = config["gradient_clipping"] if "gradient_clipping" in config else False,
            cycle_consistent_warmup = config["cycle_consistent_warmup"] if "cycle_consistent_warmup" in config else 0, 
            cycle_consistent_beta = config["cycle_consistent_beta"] if "cycle_consistent_beta" in config else 0.0,
            data_manager = data_manager,
            model = model
        )

    """
        Setup function for initializing optimizers
    """
    def _setup_optimizers(self):
        # default optimizer takes all parameters
        if self.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.999), lr=self.lr, weight_decay=self.weight_decay) # todo tune adam parameters
    """
        Sets up a wrapper around the triplet dataset that does hard triplet mining
    """
    def _setup_triplet_mining(self):
        self.data_manager.triplet_train = TripletMiningDatasetWrapper(self.model, self.data_manager.triplet_train)
        self.data_manager.setup_data_loaders(triplet_mining=True)
    
    """
       Save the combined_train_loss array in the given file 
    """
    def save_combined_loss(self, dir_file_path):
        loss_object = {
            "train": self.combined_train_loss,
            "test": self.combined_test_loss
        }
        loss_path = dir_file_path+"/"+"loss.pkl"
        with open(loss_path, "wb") as f:
            pickle.dump(loss_object, f)
    
    """
        Load the combined loss object from the given path
    """
    def load_combined_loss(self, dir_file_path):
        loss_path = dir_file_path+"/"+"loss.pkl"
        if os.path.exists(loss_path):
            with open(loss_path, "rb") as f:
                loss_object = pickle.load(f)
                self.combined_train_loss = loss_object["train"]
                self.combined_test_loss = loss_object["test"]
    
    """
        Performs forward pass on a triplet
    """
    def triplet_forward(self, triplet):
        anchor_x, positive_x, negative_x = triplet
        # put on gpu
        anchor_x = anchor_x.to(self.device)
        positive_x = positive_x.to(self.device)
        negative_x = negative_x.to(self.device)
        # run forward passes on the triplet data
        anchor_mean, anchor_logvar, _, _ = self.model(anchor_x)
        positive_mean, positive_logvar, _, _ = self.model(positive_x)
        negative_mean, negative_logvar, _, _ = self.model(negative_x)

        return ((anchor_mean, anchor_logvar), (positive_mean, positive_logvar), (negative_mean, negative_logvar))
    
    def log_free_memory(self):
        summary = torch.cuda.memory_summary(device=0)   

    """
        Run additional test metrics on the learned model
    """
    def test_additional_metrics(self):
        if isinstance(self.model, MaskedVAE) or isinstance(self.model, MaskedVAEIsolated) or isinstance(self.model, IsolatedVAE):
            self.model.similarity_mode = True 
        # test percentage of queries satisfied
        # test percentage of points consistent with a subset of queries over time
        # test reconstruction error of metadata statistics
        try:
            training_test.response_model_probability(self.model, self.data_manager.triplet_test, train=False, use_basic_setting=True)
        except:
            traceback.print_exc()

        try:
            reconstruction_of_metadata(self.model, self.data_manager.image_test, self.data_manager.metadata_test) 
        except:
            traceback.print_exc()

        # get morpho mnist spearmans
        """
        try:
            # training_test.get_morpho_mnist_spearmans(self.model, self.data_manager)
        except:
            traceback.print_exc()
        """
        # isolated vae
        if isinstance(self.model, MaskedVAE) or isinstance(self.model, MaskedVAEIsolated) or isinstance(self.model, IsolatedVAE):
            self.model.similarity_mode = False 

    """
        Tests the model for one epoch
    """
    def test_epoch(self, epoch=0):
        test_loader = self.data_manager.image_test_loader
        triplet_test_loader = self.data_manager.triplet_test_loader
        # loop will iterate until the shorter of the two iterators is exausted
        recon_iter = iter(test_loader)
        triplet_iter = iter(triplet_test_loader)
        # set loss function to test mode
        self.model.training = False
        if self.data_manager.triplet_test.indexed:
            self.model.loss_function.train_mode = False 

        if isinstance(self.model.loss_function, BetaTCVAELoss):
            self.model.loss_function.num_iters = 0
            self.model.loss_function.dataset_size = len(self.data_manager.image_test)
        cumulative_loss_dict = None
        num_iterations = 0
        for (recon_data, triplet_data) in zip(recon_iter, triplet_iter): 
            # load reconstruction data
            x = recon_data
            # load triplet data
            (anchor_x, positive_x, negative_x, attribute_index), _ = triplet_data
            triplet_input_data = (anchor_x, positive_x, negative_x, attribute_index)
            # handle basic reconstruction 
            x = x.to(self.device)
            mean, log_var, latent, recon = self.model(x)
            # test triplet loss no matter what
            anchor, positive, negative = self.triplet_forward((anchor_x, positive_x, negative_x))
            (anchor_x, positive_x, negative_x, attribute_index), _ = triplet_data
            triplet_data = (anchor, positive, negative, attribute_index)
            # calculate cumulative loss
            loss_dict = self.model.loss_function(x, recon, mean, log_var, triplet_data, triplet_input_data=triplet_input_data, test_mode=True, model=self.model)
            # calculate cumulative loss 
            # update parameters
            #assert "loss" in loss_dict
            #loss = loss_dict["loss"]
            # log the loss dict
            #del loss
            cumulative_loss_dict = add_dicts(loss_dict, cumulative_loss_dict)
            num_iterations += 1
        # run additional tests
        self.test_additional_metrics()
        loss_dict = divide_dict(cumulative_loss_dict, num_iterations)
        self.combined_test_loss.append(loss_dict)
        loss = loss_dict["loss"]
        self.test_loss.append(loss) 
        for key in loss_dict.keys():
            wandb.log({"test_"+key:loss_dict[key], "epoch": epoch})
 
        return loss
        
    """
        Trains the model for one epoch
    """
    def train_epoch(self, epoch=1):
        self.model.train()
        if self.mcmv_sampling:
            self.data_manager.triplet_train.mcmv_sampling = True
            self.data_manager.triplet_train.model = self.model
        # setup loaders
        image_train_loader = self.data_manager.image_train_loader
        triplet_train_loader = self.data_manager.triplet_train_loader
        # loop will iterate until the shorter of the two iterators is exausted
        recon_iter = iter(image_train_loader)
        triplet_iter = iter(triplet_train_loader)
        # set loss function to test mode
        if self.data_manager.triplet_train.indexed:
            self.model.loss_function.train_mode = True 
        # step scheduler
        use_scheduler = True
        if use_scheduler:
            self.scheduler.step()

        if epoch < self.cycle_consistent_warmup:
            self.model.loss_function.cycle_consistent_beta = 0.0
        else:
            self.model.loss_function.cycle_consistent_beta = self.cycle_consistent_beta

        if self.reparam_lambda < 1.0:
            self.reparam_lambda += (1/self.lambda_warmup_epochs)
        self.model.reparam_lambda = self.reparam_lambda
        # save losses
        if isinstance(self.model.loss_function, BetaTCVAELoss):
            self.model.loss_function.num_iters = 0
            self.model.loss_function.dataset_size = len(self.data_manager.image_train)
            # recon warm start
        if type(self.model.loss_function).__name__ is BayesianTripletLoss.__name__:
            self.model.loss_function.epoch = epoch

        if epoch <= self.recon_zero_start:
            self.model.loss_function.recon_beta = 0.0
        elif epoch <= self.recon_warm_start + self.recon_zero_start:
            self.model.loss_function.recon_beta = (epoch - self.recon_zero_start) * (self.recon_beta / self.recon_warm_start)
        # update kl beta
        if not self.kl_max is -1:
            self.model.loss_function.kl_beta = min(self.kl_max, self.model.loss_function.kl_beta + self.annealing_amount)

        cumulative_loss_dict = None
        num_iterations = 0
        for (recon_data, triplet_data) in tqdm(zip(recon_iter, triplet_iter)): 
            # load reconstruction data
            x = recon_data
            # load triplet data
            # check if dataset is indexed
            (anchor_x, positive_x, negative_x, attribute_index), _ = triplet_data
            triplet_input_data = (anchor_x, positive_x, negative_x, attribute_index)
            # zero_grad
            self.optimizer.zero_grad()
            # handle basic reconstruction 
            x = x.to(self.device)
            # run a reconstruction forward pass
            mean, log_var, latent, recon = self.model(x)
            # triplet forward pass
            anchor, positive, negative = self.triplet_forward((anchor_x, positive_x, negative_x))
            triplet_data = (anchor, positive, negative, attribute_index) 
            kwargs = {"model":self.model}
            # calculate cumulative loss 
            loss_dict = self.model.loss_function(x, recon, mean, log_var, triplet_data, triplet_input_data=triplet_input_data, test_mode=False, model=self.model)
            # update parameters
            assert "loss" in loss_dict
            loss = loss_dict["loss"]
            loss.backward()
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
            self.optimizer.step()
            # add to cumulative dict
            # log the loss dict
            del loss
            cumulative_loss_dict = add_dicts(cumulative_loss_dict, loss_dict)
            num_iterations += 1
        
        loss_dict = divide_dict(cumulative_loss_dict, num_iterations)
        self.combined_train_loss.append(loss_dict)
        loss = loss_dict["loss"]
        self.train_loss.append(loss) 
        for key in loss_dict.keys():
            wandb.log({"train_"+key:loss_dict[key], "epoch": epoch})

        return loss

    def save_best_model(self):
        # make save dir
        save_path = os.path.dirname(os.path.dirname(__file__))+"/logs"
        run_name = wandb.run.name
        save_dir_name = save_path+"/"+run_name
        # serialize model
        model_path = save_dir_name + "/best_model.pkl"
        torch.save(self.model.state_dict(), model_path)

    """
        Trains the model given the train_loader and test_loader
    """
    @wandb_mixin
    def train(self, epochs=1):
        self.epochs = epochs
        # setup the scheduler
        lowest_loss = None
        self.scheduler = self._setup_scheduler(self.epochs)
        print("Starting training ...")
        with torch.autograd.detect_anomaly():
            for epoch in tqdm(range(1, epochs+1), position=0, leave=True):
                # training epoch
                train_loss = self.train_epoch(epoch=epoch)
                # testing epoch
                test_loss = self.test_epoch(epoch=epoch)
                # save if lowest
                if lowest_loss is None:
                    lowest_loss = test_loss
                if test_loss <= lowest_loss:
                    lowest_loss = test_loss
                    self.save_best_model() 

        print('training complete.')
       
