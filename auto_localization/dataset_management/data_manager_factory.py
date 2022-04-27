from auto_localization.dataset_management.data_manager import DataManager
from auto_localization.oracles.metadata_oracle import MetadataOracle
from auto_localization.oracles.indexed_metadata_oracle import IndexedMetadataOracle
from auto_localization.oracles.indexed_class_oracle import IndexedClassOracle
from auto_localization.oracles.oracle import EnsembleOracle
from auto_localization.oracles.vggface_oracle import VGGFaceOracle
from auto_localization.oracles.resnet_oracle import ResnetOracle
from auto_localization.oracles.matrix_oracle import MatrixOracle
# import datasets
import datasets.morpho_mnist.dataset as morpho_mnist
import datasets.celeba.dataset as celeba
import datasets.dsprites.dataset as dsprites
import os

"""
    Function that takes in a dataset config and generates a data manager
    using the information from it
"""
def construct_data_manager(dataset_config):
    # get dataset name
    if "dataset_name" in dataset_config:
        dataset_name = dataset_config["dataset_name"]
    else:
        dataset_name = "MorphoMNIST"
    # choose the correct data manager constructor based on name
    if dataset_name == "CelebA":
        data_manager = construct_celeb_a(dataset_config)
    elif dataset_name == "MorphoMNIST":
        data_manager = construct_morpho_mnist(dataset_config)
    elif dataset_name == "dSprites":
        data_manager = construct_dsprites(dataset_config)
    else:
        raise Exception("Unrecognized dataset name : {}".format(dataset_name))
    
    return data_manager

def construct_dsprites(dataset_config):
    num_workers = dataset_config["num_workers"] if "num_workers" in dataset_config else 4
    if "localization_component_weighting" in dataset_config:
        localization_component_weighting = dataset_config["localization_component_weighting"]
    
    component_weighting = dataset_config["component_weighting"]
    batch_size = dataset_config["batch_size"]
    triplet_batch_size = dataset_config["triplet_batch_size"] if "triplet_batch_size" in dataset_config else batch_size
    attribute_return = dataset_config["attribute_return"]
    indexed = dataset_config["indexed"]
    single_feature_triplet = dataset_config["single_feature_triplet"]
    inject_triplet_noise = 0.0 if not "inject_triplet_noise" in dataset_config else dataset_config["inject_triplet_noise"]
    # make metadata oracle as trian oracle
    metadata_dataset = dsprites.MetadataDataset(
        train=True, 
    )
    test_metadata_dataset = dsprites.MetadataDataset(
        train=False, 
    )
    metadata_oracle = IndexedMetadataOracle(
        metadata_dataset=metadata_dataset,
        component_weighting=component_weighting,
        inject_triplet_noise=inject_triplet_noise
    )
    test_metadata_oracle = IndexedMetadataOracle(
        metadata_dataset=test_metadata_dataset,
        component_weighting=component_weighting,
    )
    # setup data manager
    print("Setting up data")
    triplet_train_dataset = dsprites.TripletDataset(
        train=True,
        oracle=metadata_oracle,
        attribute_return=attribute_return, 
        single_feature_triplet=single_feature_triplet, 
        inject_triplet_noise=inject_triplet_noise,
    ) 
    triplet_test_dataset = dsprites.TripletDataset(
        train=False, 
        oracle=test_metadata_oracle, 
        attribute_return=attribute_return,
        single_feature_triplet=single_feature_triplet,
    )
    image_train_dataset = dsprites.ImageDataset(
        train=True,
    )
    image_test_dataset = dsprites.ImageDataset(
        train=False,
    )
    data_manager = DataManager(
        (image_train_dataset, image_test_dataset), 
        (triplet_train_dataset, triplet_test_dataset), 
        (metadata_dataset, test_metadata_dataset), 
        batch_size=batch_size, 
        triplet_batch_size=triplet_batch_size
    )
    
    localization_metadata_oracle = MetadataOracle(
        metadata_dataset=test_metadata_dataset,
        component_weighting=component_weighting,
    )

    return data_manager, localization_metadata_oracle
   
def construct_morpho_mnist(dataset_config):
    num_workers = dataset_config["num_workers"] if "num_workers" in dataset_config else 4
    if "localization_component_weighting" in dataset_config:
        localization_component_weighting = dataset_config["localization_component_weighting"]
    if "apply_transform" in dataset_config:
        apply_transform = dataset_config["apply_transform"]
    else:
        apply_transform = False
    if "fixed_triplet" in dataset_config:
        fixed_triplet = dataset_config["fixed_triplet"]
    else:
        fixed_triplet = False    
    
    which_digits = dataset_config["which_digits"]
    component_weighting = dataset_config["component_weighting"]
    batch_size = dataset_config["batch_size"]
    triplet_batch_size = batch_size if not "triplet_batch_size" in dataset_config else dataset_config["triplet_batch_size"]
    one_two_ratio = dataset_config["one_two_ratio"]
    attribute_return = dataset_config["attribute_return"]
    indexed = dataset_config["indexed"]
    single_feature_triplet = dataset_config["single_feature_triplet"]
    inject_triplet_noise = 0.0 if not "inject_triplet_noise" in dataset_config else dataset_config["inject_triplet_noise"]
    print("inject")
    print(inject_triplet_noise)
    # make metadata oracle as trian oracle
    metadata_dataset = morpho_mnist.MetadataDataset(
        train=True, 
        which_digits=which_digits
    )
    test_metadata_dataset = morpho_mnist.MetadataDataset(
        train=False, 
        which_digits=which_digits
    )
    metadata_oracle = morpho_mnist.IndexedMetadataOracle(
        metadata_dataset=metadata_dataset,
        component_weighting=component_weighting,
        inject_triplet_noise=inject_triplet_noise
    )
    test_metadata_oracle = morpho_mnist.IndexedMetadataOracle(
        metadata_dataset=test_metadata_dataset,
        component_weighting=component_weighting,
    )
    # setup data manager
    print("Setting up data")
    triplet_train_dataset = morpho_mnist.TripletDataset(
        train=True,
        which_digits=which_digits,
        oracle=metadata_oracle,
        one_two_ratio=one_two_ratio,
        attribute_return=attribute_return, 
        single_feature_triplet=single_feature_triplet, 
        inject_triplet_noise=inject_triplet_noise,
    ) 
    triplet_test_dataset = morpho_mnist.TripletDataset(
        train=False, 
        which_digits=which_digits, 
        oracle=test_metadata_oracle, 
        one_two_ratio=one_two_ratio, 
        attribute_return=attribute_return,
        single_feature_triplet=single_feature_triplet,
    )
    image_train_dataset = morpho_mnist.ImageDataset(
        train=True,
        which_digits=which_digits
    )
    image_test_dataset = morpho_mnist.ImageDataset(
        train=False,
        which_digits=which_digits
    )
    data_manager = DataManager(
        (image_train_dataset, image_test_dataset), 
        (triplet_train_dataset, triplet_test_dataset), 
        (metadata_dataset, test_metadata_dataset), 
        batch_size=batch_size, 
        triplet_batch_size=triplet_batch_size
    )
    
    localization_metadata_oracle = morpho_mnist.MetadataOracle(
        metadata_dataset=test_metadata_dataset,
        component_weighting=component_weighting,
    )

    """
    if indexed: 
        localization_metadata_oracle = morpho_mnist.IndexedMetadataOracle(
            metadata_dataset=test_metadata_dataset,
            component_weighting=component_weighting,
        )
    else:
        localization_metadata_oracle = MetadataOracle(
            metadata_dataset=test_metadata_dataset, 
            component_weighting=component_weighting
        )
    """
    #return data_manager, localization_metadata_oracle
    return data_manager, localization_metadata_oracle

def construct_celeb_a(dataset_config):
    which_attributes = dataset_config["which_attributes"]
    batch_size = dataset_config["batch_size"]
    input_shape = dataset_config["input_shape"]
    num_workers = dataset_config["num_workers"] if "num_workers" in dataset_config else 4
    img_size = input_shape
    oracle_matrix_path = dataset_config["oracle_matrix_path"] 
    test_oracle_matrix_path = dataset_config["test_oracle_matrix_path"] 
    oracle_type = dataset_config["oracle_type"]
    single_feature_triplet = dataset_config["single_feature_triplet"]
    # make metadata oracle as trian oracle
    metadata_dataset = celeba.CelebAMetadataDataset(
            train=True, 
    )
    test_metadata_dataset = celeba.CelebAMetadataDataset(
            train=False, 
    )
    # setup data manager
    train_oracle = MatrixOracle(path=oracle_matrix_path)
    triplet_train_dataset = celeba.TripletDataset(
        train=True, 
        which_attributes=which_attributes,
        oracle=train_oracle, 
        single_feature_triplet=single_feature_triplet,
        input_shape=img_size
    )
    if oracle_type == "VGGFace":
        test_oracle = MatrixOracle(path=test_oracle_matrix_path)
        #test_oracle = VGGFaceOracle(path=test_oracle_matrix_path)
    elif oracle_type == "Resnet":
        test_oracle = MatrixOracle(path=test_oracle_matrix_path)
    elif oracle_type == "Facenet":
        test_oracle = MatrixOracle(path=test_oracle_matrix_path)
    else:
        raise NotImplementedError("Unrecognized oracle type")
    triplet_test_dataset = celeba.TripletDataset(
        train=False, 
        which_attributes=which_attributes,
        oracle=test_oracle, 
        single_feature_triplet=single_feature_triplet,
        input_shape=img_size
    )
    image_train_dataset = celeba.ImageDataset(
        train=True,
        input_shape=img_size
    )
    image_test_dataset = celeba.ImageDataset(
        train=False,
        input_shape=img_size
    )
    data_manager = DataManager((image_train_dataset, image_test_dataset), (triplet_train_dataset, triplet_test_dataset), (metadata_dataset, test_metadata_dataset), batch_size=batch_size, num_workers=num_workers)

    return data_manager, test_oracle
