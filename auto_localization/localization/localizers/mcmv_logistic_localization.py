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

################################################# Active Localizer Classes ###################################################

logistic_model = """
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

"""
    Localization system based on MCMV
"""
class MCMVLogisticLocalizer(ActiveLocalizer):
    
    def __init__(self, normalization=KNormalizationType.CONSTANT, stan_file="logistic_model.pkl", lambda_pen_MCMV=1.0):
        super().__init__(stan_file=stan_file)
        self.normalization = normalization
        self.queries = []
        self.mu_W = 0
        self.errors = []
        self.mode = "MCMV method"
        self.lambda_pen_MCMV = lambda_pen_MCMV
        self.stan_file = stan_file

    def initialize(self, **kwargs):
        super().initialize(**kwargs)
        self.initialize_stan()
        self.generate_embedding(variances=True) # generate the embedding and save the variances

    def initialize_stan(self):
        # load the stan model
        if os.path.exists(self.stan_file):
            # laod the existing model 
            self.sm = pickle.load(open(self.stan_file, 'rb'))
        else:
            model_string = logistic_model
            # make the model and save it
            self.sm = pystan.StanModel(model_code=model_string)
            with open(stan_file, 'wb') as f:
                pickle.dump(self.sm, f)

    """
        Gets a query based on the previous query responses and the 
        embedding space.
    """
    def get_query(self, sample_from_existing=True):
        """ 
            Returns W_samples from the posterior if it is defined 
        """ 
        def get_posterior_samples():
            # given measurements 0..i, get posterior samples
            if not self.A:
                # take a random sample from the embedding
                num_items = np.shape(self.embedding)[0]
                indices = np.random.randint(0, high=num_items, size=self.Nsamples)
                W_samples = self.embedding[indices]
            else:
                data_gen = {
                    'D': self.D,
                    'k': self.k,
                    'M': len(self.A),
                    'A': self.A,
                    'tau': self.tau,
                    'y': self.choices,
                    'bounds': self.bounds,
                    'mean_val': self.mean, 
                    'std': self.std
                }
                # num_samples = iter * chains / 2, unless warmup is changed
                fit = ActiveLocalizer.sm.sampling(
                    data=data_gen, 
                    iter=self.Niter, 
                    chains=int((self.Nsamples*2)/self.Niter), 
                    init=0
                )
                W_samples = fit.extract()['W']

            return W_samples

        """
            Returns the optimal mean cut max variance query
            from a chosen sample
        """
        def get_mcmv_query(sample):
            # scale the embedding
            scaled_embedding = self.embedding * self.similarity_weight
            # core mcmv functionality
            Wcov = np.cov(W_samples, rowvar=False)
            self.vars.append(Wcov)
            mcmv_values = np.zeros((self.Npairs,))
            for j in range(self.Npairs):
                p = sample[j]
                index_a, index_b = p
                (A_emb, tau_emb) = pair2hyperplane(p, scaled_embedding, self.normalization)
                varest = np.dot(A_emb, Wcov).dot(A_emb)
                distmu = np.abs(
                    (np.dot(A_emb, self.mu_W) - tau_emb)
                    / np.linalg.norm(A_emb)
                )

                mcmv_values[j] = self.k * np.sqrt(varest) - self.lambda_pen_MCMV * distmu
            # choose the best one
            optimal_query_pair = sample[np.argmax(mcmv_values)] 
            return optimal_query_pair

        # get posterior samples
        W_samples = get_posterior_samples()
        if W_samples.ndim < 2:
            W_samples = W_samples[:, np.newaxis]
        # Posterior sample mean
        self.mu_W = np.mean(W_samples, 0)
        self.posterior_means.append(self.mu_W)
        # generate and evaluate a batch of proposal pairs
        random_query_pairs = self.get_random_pairs(self.N, self.Npairs)
        optimal_query_pair = get_mcmv_query(random_query_pairs)
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

