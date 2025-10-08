# data.py


# ================================================================================================== #
#   This script contains the functions to extract the aggregated data and create data loader         #
# ================================================================================================== #


import numpy as np
import os
import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
import torch
import polars as pl

# ----- Load data ----- #
def preprocess(tax_level:str="Phylum", 
               agg_abundance_dir:str="",
               metadata_dir:str="",
               samples_threshold:int=5000) -> tuple[DataFrame, DataFrame]:
    """Extract and process abundance data and metadata. The abundance data is aggregated by taxonomic level.
    Args:
        tax_level: One of 'Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', or 'All' (unaggregated data)
        agg_abundance_dir: directory of the aggregation/ folder
        metadata_dir: directory of the metadata folder
        samples_threshold (optional): Samples with reads fewer than this threshold will be removed.
    Returns:
        abundance:  [n_features, n_samples]
        metadata: shape [n_samples, ...]
    """
    
    file_path = os.path.join(agg_abundance_dir, f"aggASV_{tax_level}.tsv")
    abundance = pd.read_csv(file_path, sep='\t').set_index(tax_level).dropna()
    file_path = os.path.join(metadata_dir, "AGP.metadata.matched.tsv")
    metadata = pd.read_csv(file_path, sep="\t", header=0, low_memory=False)
    
    print(f"Metadata shape: {metadata.shape}\n"
          f"Abundance data aggregated by [{tax_level}] shape: n_features={abundance.shape[0]}, n_samples={abundance.shape[1]}")

    # Remove samples with fewer than `samples_threshold` reads.
    total_reads_per_sample = abundance.sum(axis=0)
    # print(f"2.5% quantile: {np.quantile(total_reads_per_sample, 0.025).round(2)}")
    # print(f"5% quantile: {np.quantile(total_reads_per_sample, 0.05).round(2)}")
    # print(f"10% quantile: {np.quantile(total_reads_per_sample, 0.1).round(2)}")
    abundance = abundance.loc[:,total_reads_per_sample>samples_threshold]

    # We don't remove features since the number of features are already small after aggregation

    # Convert absolute counts into relative abundance (sum to 100).
    abundance = abundance.apply(lambda x: x/x.sum()*100, axis=0)

    # Realign metadata and abundance data
    sample_ids_metadata = np.array(metadata['#SampleID'].values)
    sample_ids_abundance = np.array(abundance.columns)
    sample_ids_joint = np.intersect1d(sample_ids_metadata, sample_ids_abundance, assume_unique=True)
    metadata = metadata[metadata['#SampleID'].isin(sample_ids_joint)]
    print(f"Processed metadata shape: {metadata.shape}\n"
          f"Processed abundance data aggregated by [{tax_level}] shape: n_features={abundance.shape[0]}, n_samples={abundance.shape[1]}")
    return abundance, metadata


# ----- Create abundance dataset ----- #
class AbundanceDataset(Dataset):
    """Create a PyTorch Dataset for the abundance data
    Args:
        abundance: abundance dataframe, (n_features, n_samples), with row ids being the taxonomic name
    """
    def __init__(self, abundance:DataFrame, transformation:str|None=None):
        x = torch.tensor(abundance.values.T).to(torch.float32) # (n_samples, n_features)
        if transformation is None:
            self.features = x
        if transformation == "clr":
            x += 0.5 # pseudocount
            log_x = torch.log(x)
            self.features = log_x - torch.mean(log_x, dim=-1, keepdim=True)

        
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    def sample(self, n_samples, seed:int=123):
        """Randomly sample from the abundance data
        Args:
            n_samples: number of samples to generate
            seed: random seed
        """
        torch.manual_seed(seed)
        shuffled_indices = torch.randperm(self.features.size(0))
        sample_indices = shuffled_indices[:n_samples]
        return self.features[sample_indices]
    
    @property
    def dim(self):
        """Return the dimension (n_features) of the abundance data"""
        return self.features.shape[1]

    
# ----- Create abundance data loader ----- #
class AbundanceLoader():
    """Create a PyTorch DataLoader for the abundance data"""
    def __init__(self, batch_size:int=2, shuffle:bool=True, drop_last:bool=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
       
    def create_loader(self, abundance_dataset:Dataset) -> DataLoader:
        return DataLoader(abundance_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=self.shuffle, 
                          drop_last=self.drop_last)

