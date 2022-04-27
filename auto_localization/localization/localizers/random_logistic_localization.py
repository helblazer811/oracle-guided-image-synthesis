# sys path
import sys
sys.path.append('../../../')
import numpy as np
import scipy.special as sc
import scipy as sp
import pystan
import pickle
import pandas as pd
from enum import Enum
from auto_localization.localization.localizers.cvxopt_localize import localize
from auto_localization.localization.localizers.cvxopt_localize import ComparisonData
from matplotlib import pyplot as plt
import torch
from PIL import Image
from numpy import asarray
import random
import os
import wandb
from auto_localization.localization.localizers.active_localization import ActiveLocalizer, pair2hyperplane, KNormalizationType
from auto_localization.localization.localizers.utils import suppress_stdout_stderr

traditional_model = """
    data {  
        int<lower=0> D;             // space dimension
        int<lower=0> M;             // number of measurements so far
        real k;                     // logistic noise parameter (scale)
        vector[2] bounds;           // hypercube bounds [lower,upper]
        vector[D] std;
        vector[D] mean_val;
        int y[M];                   // measurement outcomes
        vector[D] A[M];             // hyperplane directions
        vector[M] tau;              // hyperplane offsets
    }
    parameters {
        vector<lower=bounds[1],upper=bounds[2]>[D] W;         // the user point
    }
    transformed parameters {
        vector[M] z;
        for (i in 1:M)
            z[i] = dot_product(A[i], W) - tau[i];
    }
    model {
        // prior
        W ~ normal(mean_val, std);
        // W ~ uniform(bounds[1],bounds[2]);
        // linking observations
        y ~ bernoulli_logit(k * z);
    }
"""

traditional_model = """
    data {  
        int<lower=0> D;             // space dimension
        int<lower=0> M;             // number of measurements so far
        real k;                     // logistic noise parameter (scale)
        vector[2] bounds;           // hypercube bounds [lower,upper]
        vector[D] std;
        vector[D] mean_val;
        int y[M];                   // measurement outcomes
        vector[D] A[M];             // hyperplane directions
        vector[M] tau;              // hyperplane offsets
    }
    parameters {
        vector<lower=bounds[1],upper=bounds[2]>[D] W;         // the user point
    }
    transformed parameters {
        vector[M] z;
        for (i in 1:M)
            z[i] = dot_product(A[i], W) - tau[i];
    }
    model {
        // prior
        W ~ normal(mean_val, std);
        // W ~ uniform(bounds[1],bounds[2]);
        // linking observations
        y ~ bernoulli_logit(k * z);
    }
"""


################################################# Active Localizer Classes ###################################################
"""
    Random query selection localizaer with logistic response model
"""
class RandomLogisticLocalizer(ActiveLocalizer):
    
    def __init__(self, normalization=KNormalizationType.CONSTANT, stan_file="traditional_model.pkl"):
        super().__init__(stan_file=stan_file)
        self.stan_file = stan_file
        self.normalization = normalization
        self.queries = []
        self.mu_W = 0
        self.errors = []
        self.mode = "Random Logistic Method"

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.initialize_stan()
        print("initlaizing mcmvmu")
        self.generate_embedding(variances=True) # generate the embedding and save the variances

    def initialize_stan(self):
        # load the stan model
        if os.path.exists(self.stan_file):
            # laod the existing model 
            self.sm = pickle.load(open(self.stan_file, 'rb'))
        else:
            model_string = traditional_model
            # make the model and save it
            self.sm = pystan.StanModel(model_code=model_string)
            with open(self.stan_file, 'wb') as f:
                pickle.dump(self.sm, f)

    def get_posterior_samples(self):
        # given measurements 0..i, get posterior samples
        if not self.A:
            print("no self A")
            #W_samples = np.random.uniform(
            #    self.bounds[0], self.bounds[1], (self.Nsamples, self.D))
            # take a random sample from the embedding
            num_items = np.shape(self.embedding)[0]
            indices = np.random.randint(0, high=num_items, size=self.Nsamples)
            W_samples = self.embedding[indices]
        else:
            num_queries = len(self.posterior_means)
            k = self.k * self.k_relaxation ** num_queries
            data_gen = {
                'D': self.D,
                'k': k,
                'M': len(self.A),
                'A': self.A,
                'tau': self.tau,
                'y': self.choices,
                'bounds': self.bounds,
                'mean_val': self.mean, 
                'std': self.std
            }
            with suppress_stdout_stderr():
                # num_samples = iter * chains / 2, unless warmup is changed
                fit = ActiveLocalizer.sm.sampling(data=data_gen, iter=self.Niter, refresh=0, verbose=False,
                                       chains=int((self.Nsamples*2)/self.Niter), init=0)
                W_samples = fit.extract()['W']

        # get posterior samples
        if W_samples.ndim < 2:
            W_samples = W_samples[:, np.newaxis]
        # Posterior sample mean
        self.mu_W = np.mean(W_samples, 0)
        self.posterior_means.append(self.mu_W)
        Wcov = np.cov(W_samples, rowvar=False)
        self.vars.append(Wcov)
        print(self.mu_W)

    """
        Gets a query based on the previous query responses and the 
        embedding space.
    """
    def get_query(self):
        """ 
            Returns W_samples from the posterior if it is defined 
        """ 
        # generate and evaluate a batch of proposal pairs
        random_query_pairs = self.get_random_pairs(self.N, self.Npairs)
        optimal_query_pair = random_query_pairs[0]
        # calculate the plane information
        (A_sel, tau_sel) = pair2hyperplane(
            optimal_query_pair, self.embedding, self.normalization)
        self.A.append(A_sel)
        self.tau = np.append(self.tau, tau_sel)
        # generate images from the optimal_query_pair
        embedding_tensor = torch.tensor([self.embedding[optimal_query_pair[0], :], self.embedding[optimal_query_pair[1], :]])
        embedding_tensor = embedding_tensor.type(torch.FloatTensor)
        self.queries.append(embedding_tensor.to(self.data_manager.device).detach().cpu().numpy())
        # image_pair = self.gen_model.decode(embedding_tensor.to(self.data_manager.device)).detach().cpu().numpy()
        image_pair = self.data_manager.image_test[optimal_query_pair[0]], self.data_manager.image_test[optimal_query_pair[1]] 
        if self.indexed:
            return optimal_query_pair[0], optimal_query_pair[1]
        else:
            return image_pair

    """
        Take the queries and their answers and do the localization
    """
    def localize(self):
        return self.mu_W

