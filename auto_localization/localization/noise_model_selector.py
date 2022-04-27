from enum import Enum
import numpy as np
import torch
import wandb
from tqdm import tqdm
from scipy.optimize import basinhopping
from torch.distributions.normal import Normal
from scipy.special import logit, expit

"""
    Enum of noise constant types
"""
class KNormalizationType(Enum):
    CONSTANT = 0
    NORMALIZED = 1
    DECAYING = 2

def triplet_forward(model, triplet):
    if len(triplet) == 3:
        anchor, positive, negative = triplet
        attribute_index = -1
    elif len(triplet) == 4:
        anchor, positive, negative, attribute_index = triplet
    anchor = anchor.squeeze()
    positive = positive.squeeze()
    negative = negative.squeeze()
    anchor_mean, anchor_logvar, _, _ = model.forward(anchor.cuda())
    positive_mean, positive_logvar, _, _ = model.forward(positive.cuda())
    negative_mean, negative_logvar, _, _ = model.forward(negative.cuda())

    anchor_mean = anchor_mean.detach()
    anchor_logvar = anchor_logvar.detach()
    positive_mean = positive_mean.detach()
    positive_logvar = positive_logvar.detach()
    negative_mean = negative_mean.detach()
    negative_logvar = negative_logvar.detach()

    return (anchor_mean, anchor_logvar), (positive_mean, positive_logvar), (negative_mean, negative_logvar), attribute_index

def pair2hyperplane(positive_z, negative_z, normalization):
    A_emb = 2*(positive_z - negative_z)
    if np.linalg.norm(A_emb) == 0:
        A_emb = np.ones_like(positive_z)*0.000001

    tau_emb = (np.linalg.norm(positive_z)**2 - np.linalg.norm(negative_z)**2)

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

def calculate_logistic_probability(triplet_data, k=1.0, normalization=KNormalizationType.CONSTANT):
    anchor, positive, negative, _ = triplet_data
    anchor_z = anchor[0]
    positive_z = positive[0]
    negative_z = negative[0]

    anchor_z = anchor_z.detach().cpu().numpy().squeeze()
    positive_z = positive_z.detach().cpu().numpy().squeeze()
    negative_z = negative_z.detach().cpu().numpy().squeeze()

    def logistic_function(val):
        return 1 / (1 + np.exp(-val))
        
    A_emb, tau_emb = pair2hyperplane(positive_z, negative_z, normalization=normalization)
    probability = logistic_function(k *
        (np.dot(A_emb, anchor_z) - tau_emb)
    )

    return probability

def calculate_triplet_probability(triplet_data, margin = 0.0, k=1.0):
    anchor, positive, negative, _ = triplet_data
    muA, varA = anchor[0], anchor[1].exp()
    muA = muA.unsqueeze(0)
    varA = varA.unsqueeze(0)
    muP, varP = positive[0], positive[1].exp()
    muP = muP.unsqueeze(0)
    varP = varP.unsqueeze(0)
    muN, varN = negative[0], negative[1].exp()
    muN = muN.unsqueeze(0)
    varN = varN.unsqueeze(0)

    muA2 = muA**2
    muP2 = muP**2
    muN2 = muN**2
    varP2 = varP**2
    varN2 = varN**2

    mu = torch.sum(muP2 + varP - muN2 - varN - 2*muA*(muP - muN), dim=1)
    T1 = varP2 + 2*muP2 * varP + 2*(varA + muA2)*(varP + muP2) - 2*muA2 * muP2 - 4*muA*muP*varP
    T2 = varN2 + 2*muN2 * varN + 2*(varA + muA2)*(varN + muN2) - 2*muA2 * muN2 - 4*muA*muN*varN
    T3 = 4*muP*muN*varA
    sigma2 = torch.sum(2*T1 + 2*T2 - 2*T3, dim=1)
    sigma = sigma2**0.5

    probs = Normal(loc = mu, scale = sigma + 1e-8).cdf(-1*margin)
    """
    if torch.equal(probs.cpu(), torch.Tensor([1.0])):
        probs  = torch.Tensor([0.99999])
    if torch.equal(probs.cpu(), torch.Tensor([0.0])):
        probs  = torch.Tensor([0.00001])
    """ 
    probs = torch.clamp(probs, min=1e-6, max=1-1e-6)
    probs = probs.item()
    probs = expit(logit(probs) * k)

    return probs

"""
    parameters given a model and a set of triplets. 
"""
class NoiseModelSelector():

    def __init__(self, model=None, triplet_dataset=None, num_triplets=2000, num_settings=200, selection_parameters=["normalization", "k"], localizer_type="RandomLogistic"):
        self.localizer_type = localizer_type
        self.response_model_type = self.get_response_model_type()
        self.model = model
        self.triplet_dataset = triplet_dataset
        self.num_triplets = num_triplets
        self.num_settings = num_settings
        self.selection_parameters = selection_parameters

    def get_response_model_type(self):
        if self.localizer_type == "RandomTriplet":
            return "Triplet"
        elif self.localizer_type == "RandomLogistic":
            return "Logistic"
        elif self.localizer_type == "MCMVMUTriplet":
            return "Triplet"

    """
        Generates the different parameter settings
    """
    def generate_settings(self):
        normalization_settings = [0]
        k_settings = list(np.logspace(-3, 3, num=int(self.num_settings/3)))
        settings = []
        for normalization_setting in normalization_settings:
            for k_setting in k_settings:
                setting = {"k":k_setting, "normalization":normalization_setting}
                settings.append(setting)

        return settings
    
    """
        Evaluates "num_triplets" number of triplets and saves 
        the encoded values.
    """
    def evaluate_triplets(self, triplets=None):
        out_triplets = []
        if triplets is None:
            for i in range(self.num_triplets):
                triplet, _ = self.triplet_dataset[i]
                triplet = triplet_forward(self.model, triplet)
                out_triplets.append(triplet)
        else:
            for triplet in triplets:
                triplet = triplet_forward(self.model, triplet)
                out_triplets.append(triplet)
        return out_triplets

    """
        Given a dictionary of parameters chooses the mean probability 
        of choosing the anchor for each triplet.
    """
    def compute_mean_success_probability(self, parameters, triplets):
        probabilities = self.compute_success_probabilities(parameters, triplets)
        return np.mean(probabilities)

    """
        Given a dictionary of parameters chooses the mean probability 
        of choosing the anchor for each triplet.
    """
    def compute_success_probabilities(self, parameters, triplets):
        k = parameters["k"]
        normalization = parameters["normalization"]
        probabilities = []
        for triplet in triplets:
            if self.response_model_type is "Logistic":
                probability = calculate_logistic_probability(triplet, k=k, normalization=normalization)
            elif self.response_model_type is "Triplet":
                probability = calculate_triplet_probability(triplet, k=k)
            else:
                raise Exception(f"Unrecognized response model type {self.response_model_type}")
            probabilities.append(probability)

        return probabilities

    def perform_brute_force(self, triplets):
        # generate the settings
        settings = self.generate_settings()
        # for each setting
        success_probabilities = []
        for setting in tqdm(settings):
            # compute the mean success probability
            success_probability = self.compute_mean_success_probability(setting, triplets)            
            success_probabilities.append(success_probability)
        # choose the best setting
        best_setting = settings[np.argmax(success_probabilities)]
        try:
            wandb.log({"best_setting": best_setting})
            wandb.log({"success_probability": np.amax(success_probabilities)})
        except:
            print("No wandb initialized")
        return best_setting

    def perform_basinhopping(self, triplets):
        normalization = 0
        initial_k_value = np.array([1.0])
        # choose the best setting
        def evaluate_probability(k):
            probabilities = []
            for triplet in triplets:
                anchor_z, positive_z, negative_z = triplet
                if self.response_model_type == "Logistic":
                    probability = calculate_logistic_probability(triplet_data, k=k, normalization=normalization)
                elif self.response_model_type == "Triplet":
                    probability = calculate_triplet_probability(triplet_data, k=k)
                probabilities.append(probability)
            return -1*np.mean(probabilities)
        # perform basinhopping
        result = basinhopping(evaluate_probability, initial_k_value)
        # return the best setting
        best_setting = {"normalization":normalization, "k":result.x[0]}
        wandb.log({"best_setting": best_setting})
        wandb.log({"success_probability": result.fun * -1})

        return best_setting

    """
        This tool returns a dictionary of the relevant parameters
    """
    def select_parameters(self, optimization="brute"):
        # evaluate the triplets
        triplets = self.evaluate_triplets()
        # choose algorithm
        if optimization == "basinhopping":
            best_setting = self.perform_basinhopping(triplets)
        elif optimization == "brute":
            best_setting = self.perform_brute_force(triplets)
        else:
            raise NotImplementedError("Optimization method is not found")

        return best_setting

