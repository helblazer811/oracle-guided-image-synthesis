import sys
sys.path.append("../../../..")
from auto_localization.localization.localizers.mcmv_logistic_localization import MCMVLogisticLocalizer
from auto_localization.localization.localizers.mcmvmu_triplet_localization import MCMVMUTripletLocalizer
from auto_localization.localization.localizers.random_convex_localization import RandomConvexLocalizer
from auto_localization.localization.localizers.random_logistic_localization import RandomLogisticLocalizer
from auto_localization.localization.localizers.random_triplet_localization import RandomTripletLocalizer

"""
    Takes a localizer config and returns a localizer object
"""
def get_localizer_from_config(config):
    localizer_type = config["localizer_type"]
    normalization = 0 if not "normalization" in config else config["normalization"]
    lambda_pen_MCMV = config["lambda_pen_MCMV"] if "lambda_pen_MCMV" in config else 1.0
    lambda_latent_variance = config["lambda_latent_variance"] if "lamda_latent_varinace" in config else 1.0
    if localizer_type == "MCMVLogistic":
        localizer = MCMVLogisticLocalizer(normalization=normalization, lambda_pen_MCMV=lambda_pen_MCMV)
    elif localizer_type == "MCMVMUTriplet":
        localizer = MCMVMUTripletLocalizer()
    elif localizer_type == "RandomConvex":
        localizer = RandomConvexLocalizer(normalization=normalization)
    elif localizer_type == "RandomLogistic":
        localizer = RandomLogisticLocalizer(normalization=normalization)
    elif localizer_type == "RandomTriplet":
        localizer = RandomTripletLocalizer(normalization=normalization)
    else:
        raise Exception("Localizer not recognized {}".format(localizer_type))

    return localizer


