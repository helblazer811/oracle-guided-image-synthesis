import sys
sys.path.append("../../")
from auto_localization.models.BasicVAE import BasicVAE
from auto_localization.models.MetricVAE import MetricVAE
from auto_localization.models.IsolatedVAE import IsolatedVAE
from auto_localization.models.CelebAVAE import CelebAVAE
from auto_localization.models.DummyModel import DummyModel
from auto_localization.models.MaskedVAE import MaskedVAE
from auto_localization.models.MaskedVAEIsolated import MaskedVAEIsolated
from auto_localization.models.LearnedMaskedVAE import LearnedMaskedVAE
from auto_localization.models.CelebABetaVAE import CelebABetaVAE
from auto_localization.models.CelebAVAE import CelebAVAE
from auto_localization.models.BetaTCVAE import BetaTCVAE
from auto_localization.models.ConditionalSimilarityNetwork import ConditionalSimilarityNetwork

"""
    This is a function that returns a 
    deep learning VAE model based on a given config
"""
def get_model_from_config(model_type, model_config):
    if model_type == "BasicVAE":
        model = BasicVAE.from_config(model_config)
    elif model_type == "BetaTCVAE":
        model = BetaTCVAE.from_config(model_config)
    elif model_type == "MetricVAE":
        model = MetricVAE.from_config(model_config)
    elif model_type == "IsolatedVAE":
        model = IsolatedVAE.from_config(model_config)
    elif model_type == "CelebAVAE":
        model = CelebAVAE.from_config(model_config)
    elif model_type == "DummyModel":
        model = DummyModel.from_config(model_config)
    elif model_type == "MaskedVAE":
        model = MaskedVAE.from_config(model_config)
    elif model_type == "MaskedVAEIsolated":
        model = MaskedVAEIsolated.from_config(model_config)
    elif model_type == "LearnedMaskedVAE":
        model = LearnedMaskedVAE.from_config(model_config)
    elif model_type == "CelebABetaVAE":
        model = CelebABetaVAE.from_config(model_config)
    elif model_type == "CelebAVAE":
        model = CelebAVAE.from_config(model_config)
    elif model_type == "ConditionalSimilarityNetwork":
        model = ConditionalSimilarityNetwork.from_config(model_config)
    else:
        raise Exception("Unidenfitied model name : {}".format(model_type))

    return model
