import sys
sys.path.append("../../")
from auto_localization.experiment_management.basic_experiment import BasicExperiment
from auto_localization.dataset_management.data_manager import DataManager
from datasets.celeba.dataset import CelebAMetadataDataset, ImageDataset, TripletDataset
from oracles.vggface_oracle import VGGFaceOracle
from oracles.resnet_oracle import ResnetOracle
#from oracles.facenet_oracle import FacenetOracle
import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
from torch.nn.parameter import Parameter
from tqdm import tqdm
from torch import nn
import numpy as np
import time

def oracle_forward(oracle, image):
    if isinstance(oracle, VGGFaceOracle):
        with torch.no_grad():
            return oracle.calculate_image_model(image)
    if isinstance(oracle, ResnetOracle):
        with torch.no_grad():
            return oracle.calculate_image_resnet(image)
    if isinstance(oracle, FacenetOracle):
        with torch.no_grad():
            return oracle.calculate_image_resnet(image)

def run_tests():
    print("running tests")
    image_dataset = ImageDataset(train=True)
    oracle = ResnetOracle()

    def make_distance_matrix(num_examples=100, batch_size=20, num_workers=8, attributes=np.array([0, 1])):
        print("make distance matrix test")
        distance_matrix = PairwiseDistanceMatrix(num_examples=num_examples, attributes=attributes, batch_size=batch_size, num_workers=num_workers)
        distance_matrix.compute_pairwise_distance_matrix(image_dataset, oracle)
        path = f"distance_matrix_{num_examples}_{attributes}.pth"
        distance_matrix.serialize(path)

        return path, distance_matrix
    
    def test_serialization_same():
        print("test serialization same")
        num_examples = 120
        batch_size = 40
        num_workers = 16
        attributes = np.array([0, 1])
        distance_matrix = PairwiseDistanceMatrix(num_examples=num_examples, attributes=attributes, batch_size=batch_size, num_workers=num_workers)
        distance_matrix.compute_pairwise_distance_matrix(image_dataset, oracle)
        path = f"distance_matrix_{num_examples}_{attributes}.pth"
        distance_matrix.serialize(path)
        unserialized_distance_matrix = PairwiseDistanceMatrix(num_examples=num_examples, attributes=attributes, batch_size=batch_size, num_workers=num_workers)
        unserialized_distance_matrix.unserialize(path)

        assert torch.allclose(distance_matrix.distance_matrix, unserialized_distance_matrix.distance_matrix)

    def test_embedding_variability():
        num_tests = 100
        test_image = image_dataset[0].cuda()
        # embedd the image
        embedded_vector = oracle_forward(oracle, test_image).float()[0]
        for test_num in range(num_tests):
            test_image = image_dataset[0].cuda()
            # embedd the image
            test_embedded_vector = oracle_forward(oracle, test_image).float()[0]
            assert torch.allclose(test_embedded_vector, embedded_vector)    

    def test_correct_distances():
        num_examples = 40
        batch_size = 10
        num_workers = 16
        attributes = None
        path, distance_matrix = make_distance_matrix(num_examples=num_examples, batch_size=batch_size, num_workers=num_workers, attributes=attributes)
        num_tests = 100
        oracle.model.eval()
        for test_num in range(num_tests):
            # get a random pair of indices
            matrix_indices = np.random.choice(np.arange(num_examples), size=2)
            image_indices = [
                int(distance_matrix.indices[matrix_indices[0]]),
                int(distance_matrix.indices[matrix_indices[1]])
            ]
            # evaluate their distance using the oracle
            image_a = image_dataset[image_indices[0]].cuda()
            image_b = image_dataset[image_indices[1]].cuda()

            vector_a = oracle_forward(oracle, image_a).float()[0]
            vector_b = oracle_forward(oracle, image_b).float()[0]
            if not attributes is None:
                distance = torch.abs(vector_a[attributes] - vector_b[attributes])
            else:
                distance = torch.norm(vector_a - vector_b)
            assert torch.allclose(
                distance_matrix.distance_matrix[:, int(matrix_indices[0]), int(matrix_indices[1])].cuda(), 
                distance
            )

    test_serialization_same()
    test_embedding_variability()
    test_correct_distances()

class PairwiseDistanceMatrix(nn.Module):
    
    def __init__(self, num_examples=4096, attributes=np.array([0]), batch_size=32, num_workers=8):
        super(PairwiseDistanceMatrix, self).__init__()
        self.num_examples = num_examples
        self.attributes = attributes
        if self.attributes is None:
            num_attributes = 1
        else:
            num_attributes = np.shape(self.attributes)[0]
        self.batch_size = batch_size
        self.num_workers = num_workers
        # this contains the image_dataset indices contained in this matrix
        self.indices = Parameter(torch.zeros(self.num_examples), requires_grad=False)
        self.distance_matrix = Parameter(
            torch.zeros(num_attributes, self.num_examples, self.num_examples, dtype=torch.float, requires_grad=False)
        ) # uses 16 bit precision

    def serialize(self, path):
        # save the indices and distance matrix 
        torch.save(self.state_dict(), path)

    def unserialize(self, path):
        loaded_state_dict = torch.load(path)
        # get the shapes of indices and distance_matrix
        self.indices = Parameter(torch.zeros(loaded_state_dict["indices"].shape))
        self.distance_matrix = Parameter(torch.zeros(loaded_state_dict["distance_matrix"].shape, dtype=torch.float))
        self.load_state_dict(loaded_state_dict, strict=False)
        # infer fields from loaded parameters
        num_attributes = self.distance_matrix.shape[0]
        self.attributes = torch.arange(num_attributes)
        self.num_examples = self.indices.shape[0]

    def calculate_distance(self, index_a, index_b, attribute_index=-1):
        # first makes sure the indices are in the matrix
        #assert len(torch.where(self.indices.data == index_a)[0]) > 0 and len(torch.where(self.indices.data == index_b)[0]) > 0
        # calculate the distance 
        if attribute_index == -1:
            distance = self.distance_matrix[:, index_a, index_b]
            return torch.mean(distance)
        else:
            return self.distance_matrix[attribute_index, index_a, index_b]

    def compute_pairwise_distance_matrix(self, image_dataset, oracle):
        oracle.model.eval()
        # oracle feed forward function
        # generate a random sample of indices from the train dataset
        indices = np.random.choice(np.arange(len(image_dataset)), size=self.num_examples).astype(int)
        #indices = np.arange(self.num_examples).astype(int)
        self.indices.data = torch.Tensor(indices)
        image_dataset = torch.utils.data.Subset(image_dataset, indices)
        # generate the distance matrix
        half_batch_size = self.batch_size // 2
        # loop through each index
        i_batch_sampler = BatchSampler(SequentialSampler(image_dataset), batch_size=half_batch_size, drop_last=False)
        for i_indices in tqdm(i_batch_sampler):
            i_images = image_dataset[i_indices].cuda()
            # get the model vectors
            i_vectors = oracle_forward(oracle, i_images).cpu().float()
            # make another sampler
            j_batch_sampler = BatchSampler(SequentialSampler(image_dataset), batch_size=half_batch_size, drop_last=False)
            # iterate through the indices
            for j_indices in j_batch_sampler:
                # optimization
                if np.all(j_indices > np.max(i_indices)):
                    continue
                j_images = image_dataset[j_indices].cuda()
                # get the model vectors
                j_vectors = oracle_forward(oracle, j_images).cpu().float()
                # compute the distance
                if not self.attributes is None:
                    distance = torch.abs(i_vectors[:, None, self.attributes] - j_vectors[None, :, self.attributes])
                    distance = distance.permute(2, 0, 1)
                else:
                    distance = i_vectors[:, None, :] - j_vectors[None, :, :]
                    distance = torch.norm(distance, dim=-1).unsqueeze(-1)
                    distance = distance.permute(2, 0, 1)
                # assign to the distance matrix
                start_i = np.min(i_indices)
                end_i = np.max(i_indices) + 1
                start_j = np.min(j_indices)
                end_j = np.max(j_indices) + 1
                self.distance_matrix[:, start_i:end_i, start_j:end_j] = distance
                self.distance_matrix[:, start_j:end_j, start_i:end_i] = distance.permute(0, 2, 1)

        print("Finished matrix computation") 

if __name__ == "__main__":
    test = False
    if test:
        run_tests()
    num_examples = 20000
    batch_size = 50
    num_workers = 4
    # attributes = np.array([0, 1])
    attributes = None # distance in entire space
    train = True
    input_shape = (160, 160)
    image_dataset = ImageDataset(train=train, input_shape=input_shape)
    oracle = VGGFaceOracle()
    distance_matrix = PairwiseDistanceMatrix(num_examples=num_examples, attributes=attributes, batch_size=batch_size, num_workers=num_workers)
    distance_matrix.compute_pairwise_distance_matrix(image_dataset, oracle)
    path = f"vggface_matrix_{num_examples}_{attributes}_train_{train}.pth"
    distance_matrix.serialize(path)
