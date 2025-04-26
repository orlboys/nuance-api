import torch
from torch.utils.data import DataLoader, random_split
from .dataset import BiasDataset
from config import (
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    SHUFFLE_DATA, SEED, TRAIN_SPLIT
)

def create_dataloaders(csv_file):
    """
    Creates both training and validation DataLoaders by splitting the dataset.

    Params:
    csv_file (str): Path to the dataset CSV file.

    Returns:
    Tuple[DataLoader, DataLoader]: train_loader and val_loader.
    """
    full_dataset = BiasDataset(csv_file)
    torch.manual_seed(SEED)

    # Split the dataset
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset.augment = True  # Enable data augmentation for training set
    val_dataset.augment = False  # Disable data augmentation for validation set

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE_DATA,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return train_loader, val_loader