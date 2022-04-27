import pandas as pd 
import numpy as np
from tqdm import tqdm

"""
    Make a pandas dataframe that holds triplets based on their indices in the morpho_mnist metadata 
"""
def initialize_dataframe():
    # column names
    columns = ["anchor", "positive", "negative", "attribute_index"]
    # init dataframe
    df = pd.DataFrame(columns=columns)
    return df

def save_fixed_triplet_dataset(base_triplet_dataset, out_path, triplets_per_example=20, attribute_indices=[2,3]):
    print("Generating fixed triplet dataset")
    # init a dataframe
    df = initialize_dataframe()
    total_indices = 0
    for iteration in tqdm(range(triplets_per_example)):
        # go thruough the whole dataset and save the triplet by index in a dataframe 
        for data_index in range(len(base_triplet_dataset)):
            # get a triplet
            triplet = base_triplet_dataset.get_image_indices(data_index)
            # save the triplet to a dataframe
            anchor_index, positive_index, negative_index = triplet
            df.loc[total_indices] = [anchor_index, positive_index, negative_index]
            total_indices += 1
    # save the dataframe at path
    df.to_csv(out_path)
       

