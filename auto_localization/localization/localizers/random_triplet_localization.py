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
import wandb
from PIL import Image
from numpy import asarray
import random
import os
from torch.distributions.normal import Normal
from auto_localization.localization.localizers.active_localization import ActiveLocalizer, pair2hyperplane, KNormalizationType

triplet_model = """
    data {  
        int<lower=0> D;             // space dimension
        int<lower=0> M;             // number of measurements so far
        real k;                     // logistic noise parameter (scale)
        vector[D] prior_var;
        vector[D] prior_mean;
        vector[D] a_var[M];
        vector[D] a_mean[M];
        vector[D] b_var[M];
        vector[D] b_mean[M];
        int choices[M];                   // measurement outcomes
    }
    parameters {
        vector[D] W_mean;                // the user point
        vector<lower = 0>[D] W_var;
    }
    model {
        vector[M] probs;
        vector[D] tau_mean;
        vector[D] tau_var;
        for (i in 1:M) {
            tau_mean = square(a_mean[i]) + a_var[i] - square(b_mean[i]) - b_var[i] - 2 * W_mean .* (a_mean[i] - b_mean[i]);
            tau_var = 2 *( 
                a_var[i] .* (a_var[i] + 2 * square(a_mean[i]))+ b_var[i] .* (b_var[i] + 2*square(b_mean[i])) - 4*W_var .* a_mean[i] .* b_mean[i] 
                - 2*W_mean .* (W_mean .* (square(a_mean[i]) + square(b_mean[i])) - 2*a_mean[i] .* a_var[i] - 2*b_mean[i] .* b_var[i]) 
                + 2*(W_var + square(W_mean)).*((a_var[i] + square(a_mean[i])) + (b_var[i] + square(b_mean[i])))
            );
            probs[i] = normal_cdf(0, tau_mean, sqrt(tau_var));
        }
        // prior
        W_mean ~ normal(prior_mean, sqrt(prior_var));
        W_var ~ normal(prior_var, 1.0);
        // linking observations
        choices ~ bernoulli(probs);
    }
"""

logit_triplet_model = """
    data {  
        int<lower=0> D;             // space dimension
        int<lower=0> M;             // number of measurements so far
        real k;                     // logistic noise parameter (scale)
        vector[D] prior_var;
        vector[D] prior_mean;
        vector[D] a_var[M];
        vector[D] a_mean[M];
        vector[D] b_var[M];
        vector[D] b_mean[M];
        int choices[M];                   // measurement outcomes
    }
    parameters {
        vector[D] W_mean;                // the user point
        vector<lower = 0>[D] W_var;
    }
    model {
        vector[M] probs;
        vector[D] tau_mean;
        vector[D] tau_var;
        for (i in 1:M) {
            tau_mean = square(a_mean[i]) + a_var[i] - square(b_mean[i]) - b_var[i] - 2 * W_mean .* (a_mean[i] - b_mean[i]);
            tau_var = 2 *( 
                a_var[i] .* (a_var[i] + 2 * square(a_mean[i]))+ b_var[i] .* (b_var[i] + 2*square(b_mean[i])) - 4*W_var .* a_mean[i] .* b_mean[i] 
                - 2*W_mean .* (W_mean .* (square(a_mean[i]) + square(b_mean[i])) - 2*a_mean[i] .* a_var[i] - 2*b_mean[i] .* b_var[i]) 
                + 2*(W_var + square(W_mean)).*((a_var[i] + square(a_mean[i])) + (b_var[i] + square(b_mean[i])))
            );
            probs[i] = normal_cdf(0, tau_mean, sqrt(tau_var));
        }
        // prior
        W_mean ~ normal(prior_mean, sqrt(prior_var));
        W_var ~ normal(prior_var, 1.0);
        // logit transform
        probs = logit(probs);
        // linking observations
        choices ~ bernoulli_logit(k * probs);
    }
"""

"""
    Localization system based on MCMVMU
    This is a variation of MCMV that takes into account uncertainty quantification
"""
class RandomTripletLocalizer(ActiveLocalizer):
    
    def __init__(self, normalization=KNormalizationType.CONSTANT, stan_file="triplet_model.pkl"):
        super().__init__(stan_file=stan_file, override_stan=True)
        # declare variables
        self.normalization = normalization
        self.queries = []
        self.mu_W = 0
        self.errors = []
        self.mode = "Random Triplet Method"
        self.stan_file = stan_file

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.initialize_stan()
        self.generate_embedding(variances=True) # generate the embedding and save the variances
        # compute the prior variance and prior mean 
        self.prior_var = np.var(self.embedding, axis=0)
        self.prior_mean = np.mean(self.embedding, axis=0)
        self.posterior_means.append(self.prior_mean)
        self.a_var = []
        self.a_mean = []
        self.b_var = []
        self.b_mean = []

    def initialize_stan(self):
        # load the stan model
        if os.path.exists(self.stan_file):
            # laod the existing model 
            self.sm = pickle.load(open(self.stan_file, 'rb'))
        else:
            model_string = logit_triplet_model
            # make the model and save it
            self.sm = pystan.StanModel(model_code=model_string)
            with open(self.stan_file, 'wb') as f:
                pickle.dump(self.sm, f)

    """ 
        Returns W_samples from the posterior if it is defined 
    """ 
    def get_posterior_distribution_params(self):
        # given measurements 0..i, get posterior samples
        if len(self.choices) < 1:
            W_cov = np.cov(self.embedding.T)
            W_mean = np.mean(self.embedding, axis=0)
            return W_mean, W_cov
        else:
            # make the STAN data object            
            data_gen = {
                'D': self.D,
                'M': len(self.a_var),
                'k': self.k,
                'prior_var': self.prior_var,
                'prior_mean': self.prior_mean,
                'a_var': self.a_var,
                'a_mean': self.a_mean,
                'b_var': self.b_var,
                'b_mean': self.b_mean,
                'choices': self.choices
            }
            # num_samples = iter * chains / 2, unless warmup is changed
            fit = self.sm.sampling(data=data_gen, iter=self.Niter,
                                   chains=int((self.Nsamples*2)/self.Niter), init=0)
            # get the sample mean and sample variance                       
            W_mean = fit.extract()['W_mean']
            W_mean = np.mean(W_mean, axis=0)
            W_var = fit.extract()['W_var']
            W_var = np.mean(W_var, axis=0)
            # turn the variance vector into a covariance matrix
            W_cov = np.diag(W_var)
            return W_mean, W_cov

    """
        Gets a query based on the previous query responses and the 
        embedding space.
    """
    def get_query(self, sample_from_existing=True):
        # get posterior samples
        W_mean, W_cov = self.get_posterior_distribution_params()
        #self.log_probability_of_choosing_reference()
        self.vars.append(W_cov)
        self.mu_W = W_mean
        self.posterior_means.append(self.mu_W)
        # generate and evaluate a batch of proposal pairs
        random_query_pairs = self.get_random_pairs(self.Npairs, self.Npairs)
        optimal_query_pair = random_query_pairs[0]
        index_a, index_b = optimal_query_pair[0], optimal_query_pair[1]
        # save the variances of the best choice
        self.a_variance = np.exp(self.embedding_logvars[index_a])
        self.a_mean_current = self.embedding[index_a]
        self.b_variance = np.exp(self.embedding_logvars[index_b])
        self.b_mean_current = self.embedding[index_b]
        # use choices to select a_var, a_mean, b_var, and b_mean
        self.a_var.append(self.a_variance)
        self.a_mean.append(self.a_mean_current)
        self.b_var.append(self.b_variance)
        self.b_mean.append(self.b_mean_current)
        # generate images from the optimal_query_pair
        embedding_tensor = torch.tensor([self.embedding[optimal_query_pair[0], :], self.embedding[optimal_query_pair[1], :]])
        embedding_tensor = embedding_tensor.type(torch.FloatTensor)
        self.queries.append(embedding_tensor.to(self.data_manager.device).detach().cpu().numpy())
        # image_pair = self.gen_model.decode(embedding_tensor.to(self.data_manager.device)).detach().cpu().numpy()
        image_pair = self.data_manager.image_test[optimal_query_pair[0]], self.data_manager.image_test[optimal_query_pair[1]] 
        return image_pair

    """
        Take the queries and their answers and do the localization
    """
    def localize(self):
        return self.mu_W

