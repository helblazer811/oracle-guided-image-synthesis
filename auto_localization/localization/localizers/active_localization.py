# sys path
import multiprocessing
import sys
sys.path.append('../..')
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
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

"""
    Enum of noise constant types
"""
class KNormalizationType(Enum):
    CONSTANT = 0
    NORMALIZED = 1
    DECAYING = 2


#################################################### Helper Functions ########################################################

"""
    converts pair to hyperplane weights and bias. 
"""
def pair2hyperplane(p, embedding, normalization, slice_point=None):
    A_emb = 2*(embedding[p[0], :] - embedding[p[1], :])
    if np.linalg.norm(A_emb) == 0:
        A_emb = np.ones_like(embedding[0, :])*0.000001

    if slice_point is None:
        tau_emb = (np.linalg.norm(embedding[p[0], :])**2
                - np.linalg.norm(embedding[p[1], :])**2)
    else:
        tau_emb = np.dot(A_emb, slice_point)

    if normalization == KNormalizationType.CONSTANT:
        pass
    elif normalization == KNormalizationType.NORMALIZED:
        A_mag = np.linalg.norm(A_emb)
        A_emb = A_emb / A_mag
        tau_emb = tau_emb / A_mag
    elif normalization == KNormalizationType.DECAYING:
        A_mag = np.linalg.norm(A_emb)
        A_emb = A_emb * np.exp(-A_mag)
        tau_emb = tau_emb * np.exp(-A_mag)
    return (A_emb, tau_emb)

####################################################### Stan Model ########################################################
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

bayesian = """
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
        //W_mean ~ std_normal();
        //W_var ~ std_normal();
        // logit transform probs
        probs = logit(probs)
        // linking observations
        choices ~ bernoulli_logit(probs);
    }
"""

######################################################## Enums ##############################################################


################################################# Active Localizer Classes ###################################################

"""
    Parent class blueprint for the different localization modes
"""
class ActiveLocalizer():
    # static reference ot pystan model    
    sm = None
    def __init__(self, stan_file="model.pkl", override_stan=False):
        multiprocessing.set_start_method("fork", force=True)
        if override_stan:
            return
        # make Stan model
        if ActiveLocalizer.sm is None:
            try:
                ActiveLocalizer.sm = pickle.load(open(stan_file, 'rb'))
            except:
                print("Creating stan model because file '{}' not found".format(stan_file))
                ActiveLocalizer.sm = pystan.StanModel(model_code=traditional_model)
                with open(stan_file, 'wb') as f:
                    pickle.dump(ActiveLocalizer.sm, f)

    """
        arguments:
            embedding: np.array - an N x d embedding of points
            k: noise constant value
            normalization: model normalization scheme
            method: pair selection method

        optional arguments:
            bounds: hypercube lower and upper bounds [lb, ub]
            Nchains: number of sampling chains
            Nsamples: number of posterior samples
            pair_sample_rate: downsample rate for pair selection

            plotting settings:
                plotting: plotting flag (bool)
                plot_pause: pause time between plots, in seconds
                scale_to_embedding: if True, scale plot to embedding
                ref: np.array - d x 1 user point vector
                
            lambda_pen_MCMV: lambda penalty for MCMV method
            lambda_pen_EPMV: lambda penalty for EPMV method
        """
    def initialize(self, k=1.0, normalization=KNormalizationType.CONSTANT, gen_model=None, data=None,
                   bounds=np.array([-5, 5]), Nchains=4, Nsamples=4000, ndim=2,
                   num_pairs=100, plotting=False, ref=None, lambda_pen_MCMV=1,
                   lambda_pen_EPMV=None, k_relaxation=1.0, lambda_latent_variance=0.0, model_path=None, noise_scale=0.0, top_n_random=30, indexed=False, stan_model_type="Traditional"):
        
        self.k = k
        self.normalization = normalization
        self.ndim = ndim
        self.bounds = bounds
        self.Nchains = Nchains
        self.Nsamples = Nsamples
        self.top_n_random = top_n_random
        self.stan_model_type = stan_model_type
        self.indexed = indexed
        self.data_manager = data
        self.gen_model = gen_model
        self.k_relaxation = k_relaxation
        self.ndim = self.gen_model.z_dim
        if hasattr(self.gen_model, "similarity_dim"):
            self.ndim = self.gen_model.similarity_dim
        self.model_path = model_path
        if hasattr(self.gen_model, "loss_name") and self.gen_model.loss_name is "GlobalMaskVAETripletLoss":
            self.similarity_weight = self.gen_model.loss_function.get_mask_activation_vector()
        else:
            self.similarity_weight = np.ones(self.ndim)
        self.generate_embedding()
        self.mean = np.mean(self.embedding, axis=0)
        self.std = np.std(self.embedding, axis=0)
        self.embedded_reference = None
        self.noise_scale = noise_scale
        self.choices = []
        self.queries = []
        mins, maxs = self.get_embedding_range()
        min_val = np.amin(np.abs(mins))
        self.bounds = np.array([-min_val, min_val])
        Niter = int(2*Nsamples/Nchains)
        #assert Niter >= 1000
        self.Niter = Niter
        self.N = self.embedding.shape[0]
        self.Npairs = num_pairs 
        #self.Npairs = int(pair_sample_rate * sp.special.comb(self.N, 2))
        self.D = self.embedding.shape[1]
        self.oracle_queries_made = []
        self.mu_W = np.zeros(self.D)

        self.A = []
        self.tau = []
        self.reference_data = None
        self.posterior_means = []
        self.vars = []

        self.plotting = plotting
        self.ref = ref
        self.lambda_pen_MCMV = lambda_pen_MCMV
        self.lambda_latent_variance = lambda_latent_variance

        if lambda_pen_EPMV is None:
            self.lambda_pen_EPMV = np.sqrt(self.D)
        else:
            self.lambda_pen_EPMV = lambda_pen_EPMV

    """
        Handles skipping the current query without answering it
        because it is nonsensical. This should work because query selection
        is typically nondeterministic
    """
    def abstain_query(self):
        # remove the most recent queries
        if len(self.queries) > 0:
            del self.queries[-1]
        self.tau = self.tau[:-1]
        if len(self.A) > 0:
            del self.A[-1]

    """
        Records the users choice to a given query.
        The first query is 1 and the second one is 0. 
        Not sure why this is.
    """
    def save_choice(self, choice):
        if choice == "LEFT" or choice == 0:
            self.choices.append(1)
        elif choice == "RIGHT" or choice == 1:
            self.choices.append(0)

    def get_embedding_range(self):
        mins = np.amin(self.embedding, axis=0)
        maxs = np.amax(self.embedding, axis=0)
        
        return mins, maxs

    def get_random_pairs(self, N, M):
        embedding_size = len(self.embedding)
        indices = np.arange(embedding_size)
        indices = np.random.choice(indices, size=(N, 2))
        for i in range(indices.shape[0]):
            if indices[i, 0] == indices[i, 1]:
                indices[i, 0] += 1
                indices[i, 0] = indices[i, 0] % embedding_size

        return indices
        """
        # pair selection support function
        indices = np.random.choice(N, (int(2*M), 2))
        indices = [(i[0], i[1]) for i in indices if i[0] != i[1]]
        assert len(indices) >= M
        print(indices)
        return indices[0:M]
        """

    """
        Maps the input data to the embedding and stores it in self.embedding
    """
    def generate_embedding(self, inputs=None, variances=False):
        if not variances:
            if inputs is None:
                z_train = self.data_manager.get_latent_point_training(self.gen_model, test=True, variances=variances)
            else:
                z_train = self.data_manager.get_latent_point_training(self.gen_model, inputs=inputs, variances=variances)
            self.embedding = z_train
        else:
            if inputs is None:
                z_train, z_logvar = self.data_manager.get_latent_point_training(self.gen_model, test=True, variances=variances)
            else:
                z_train, z_logvar = self.data_manager.get_latent_point_training(self.gen_model, inputs=inputs, variances=variances)
            self.embedding = z_train
            self.embedding_logvars = z_logvar

    """
        Returns the normal vector and point representing the bisecting
        hyperplane of two queries. 
    """
    def get_linear_decision_boundaries(self):
        planes = []
        for query in self.queries:
            vector_a = query[0]
            vector_b = query[1]

            midpoint = (vector_a + vector_b)*0.5
            normal_vector = vector_a - midpoint

            planes.append((midpoint, normal))

        return planes
    
    def get_reference_image(self):
        rand_ind = None
        if self.reference_data is None:
            # choose random data from the test
            random_index = np.randint(0, len(self.data_manager.image_test))
            input_image = self.data_manager.image_test[random_index]
            self.reference_data = input_image
            self.embedded_reference = self.data_manager.gen_model.encode(input_image.to(self.data_manager.device)).detach().cpu().numpy()[0]

        return rand_ind , self.reference_data

    def get_latent_point_training(self, variances=False):
        return self.data_manager.get_latent_point_training(self.gen_model, variances=variances)

