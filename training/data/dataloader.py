from torch.utils.data import DataLoader
from .dataset import BiasDataset
from config import (
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, 
    SHUFFLE_DATA, SEED
)

"""
Config Import Info:
BATCH_SIZE: Number of samples per batch for training.
NUM_WORKERS: Number of subprocesses to use for data loading.
PIN_MEMORY: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
SHUFFLE_DATA: Whether to shuffle the dataset at every epoch.
SEED: Random seed for reproducibility.
"""

def create_dataloader(csv_file, tokenizer):
    """
    Creates a DataLoader for the BiasDataset.
    
    Params:
    csv_file (str): Path to the CSV file containing the dataset.
    tokenizer (DistilBertTokenizer): Pre-trained tokenizer for text encoding.
    
    Returns:
    DataLoader: A PyTorch DataLoader instance for the BiasDataset.
    """
    dataset = BiasDataset(csv_file, tokenizer)
    torch.manual_seed(SEED)
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_DATA,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    return dataloader