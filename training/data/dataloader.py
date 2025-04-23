from torch.utils.data import DataLoader
from .dataset import BiasDataset

def create_dataloader(csv_file, tokenizer, batch_size=32, shuffle=True, num_workers=5)
    """
    Creates a DataLoader for the BiasDataset.
    
    Params:
    csv_file (str): Path to the CSV file containing the dataset.
    tokenizer (DistilBertTokenizer): Pre-trained BERT tokenizer for text encoding.
    batch_size (int): Number of samples per batch.
    shuffle (bool): Whether to shuffle the dataset at every epoch.
    num_workers (int): Number of subprocesses to use for data loading (set to 0 for no multiprocessing).
        - NOTE: a general rule of thumb is to set num_workers to the number of CPU cores available MINUS 1.
    """
    dataset = BiasDataset(csv_file, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True, # Pin memory for faster data transfer to GPU
    )
    return dataloader