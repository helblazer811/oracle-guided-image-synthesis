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
from auto_localization.localization.localizers.active_localization import ActiveLocalizer, pair2hyperplane, KNormalizationType

################################################# Active Localizer Classes ###################################################
"""
    Localization system using a random search.
    This impliments the same functionality as what is done in ../modular_main_torch.py
"""
class RandomConvexLocalizer(ActiveLocalizer):    

    def __init__(self, normalization=KNormalizationType.CONSTANT):
        super().__init__()
        self.normalization = normalization
        self.errors = []
        self.mode = "CVX Method"

    """
        Randomly sample n latent points for testing purposes.
    """
    def generate_random_points(self,n):
        # get two random pairs from latent space
        points = []
        for i in range(n):
            points.append(torch.randn((self.ndim, 2)))
        return points

    """
        Chooses a random query from the latent space
    """
    def get_query(self):
        choose_from_existing = True
        # get two random pairs from latent space
        # W_samples = np.random.uniform(self.bounds[0], self.bounds[1], (self.Nsamples, self.D))
        if not choose_from_existing: 
            mins, maxs = self.get_embedding_range()
            embedding_range = np.linalg.norm(maxs - mins)/2
            latent_vectors = torch.randn((2, self.ndim)) * embedding_range 
            self.vars.append(np.zeros((2,2)))
            self.queries.append(latent_vectors.detach().numpy())
            image_pair = self.gen_model.decode(latent_vectors.to(self.data_manager.device)).detach().cpu().numpy()
            return image_pair
        else:
            rand_inds = np.random.choice(np.arange(np.shape(self.embedding)[0]), size=2, replace=False)
            latent_vectors = torch.Tensor(self.embedding[rand_inds])
            self.vars.append(np.zeros((2,2)))
            self.queries.append(latent_vectors.detach().numpy())
            image_pair = self.data_manager.image_test[rand_inds].squeeze().detach().cpu().numpy()
            if self.indexed:
                return rand_inds[0], rand_inds[1] 
            else:
                return image_pair
        
    """
        Records the users choice to a given query.
        The first query is 1 and the second one is 0.
        Not sure why this is.
    """
    def save_choice(self, choice):
        super(RandomConvexLocalizer, self).save_choice(choice)
        self.posterior_means.append(self.localize().squeeze())

    """
        Take the queries and their answers and do the localization
    """
    def localize(self):
        data = ComparisonData(len(self.queries), self.ndim)
        for i, query in enumerate(self.queries):
            query = query.T
            if self.choices[i] == 1:
                data.Xi[:,i] = query[:,0]
                data.Xj[:,i] = query[:,1]
            else:
                data.Xi[:,i] = query[:,1]
                data.Xj[:,i] = query[:,0]
        opt_info = localize(data)
        zhat = np.array(opt_info['x'][:self.ndim]).reshape((1,self.ndim))
        return zhat

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
