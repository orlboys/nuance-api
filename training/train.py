# =============================================================================
# TRAINING SCRIPT
# =============================================================================
# This script orchestrates the training process, including:
# - Data loading
# - Model instantiation
# - Loss computation and backpropagation
# - Evaluation and logging with TensorBoard
# - Model checkpoint saving
#
# HOW TO RUN:
#
# Activate your virtual environment:
#     ~/venv/scripts/activate
#
# Ensure you have the required libraries installed:
#     pip install -r requirements.txt
#
# Ensure your dataset is in the correct format and located at DATASET_PATH.
# The dataset should be a CSV file with two columns: 'text' and 'label'.
# The 'text' column should contain the input text data, and the 'label' column should contain the corresponding labels.
# The labels should be integers representing the classes (e.g., 0, 1, 2, etc.).
#
# Return to the root directory of the training folder (the script assumes this is the working directory, but it'll also scream at you if you run it from the wrong place):
#     cd ~/training
#
# Run this script directly to begin training:
#     python train.py
#
#
# Make sure all configuration parameters are set in `config.py`,
# and that your dataset is correctly formatted and located at DATASET_PATH.
#
# Author: orlboys (https://github.com/orlboys)
# Date: 25-04-2025
# License: MIT License
# =============================================================================
### Imports ###

# torch libraries #
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# custom libraries #
from models.bias_model import BiasModel
from data.dataset import BiasDataset
from data.dataloader import create_dataloaders
from config import (
    LEARNING_RATE, WEIGHT_DECAY, MAX_GRAD_NORM, OPTIMIZER_EPS, 
    NUM_EPOCHS, MODEL_NAME, BATCH_SIZE, USE_CUDA, DATASET_PATH,
    USE_AMP, ACCUMULATION_STEPS, CHECKPOINTS_PATH, NICKNAME,
    LOG_DIR
)

import os
import sys

# Check if the script is being run from the correct directory
if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
    print("⚠️: This script must be run from the training directory.")
    print("Please navigate to the training directory and run the script again.")
    sys.exit(1)

if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH) # Create the checkpoints directory if it doesn't exist
    print(f"✅: Checkpoint path {CHECKPOINTS_PATH} created.")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR) # Create the log directory if it doesn't exist
    print(f"✅: Log path {LOG_DIR} created.")

### GPU Configuration ###
device = torch.device("cuda" if USE_CUDA else "cpu") # Check if CUDA is available and set the device accordingly

### Data Loading ###

train_loader, val_loader = create_dataloaders(DATASET_PATH) # Create DataLoaders for training and validation datasets

### Model, Loss, Optimizer ###

model = BiasModel().to(device) # Instantiate the model with the specified model name and number of labels

loss_fn = nn.CrossEntropyLoss() # Loss function for multi-class classification (CrossEntropyLoss is suitable for multi-class classification tasks)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    eps=OPTIMIZER_EPS,
    weight_decay=WEIGHT_DECAY
)

"""
CrossEntropyLoss combines LogSoftmax and NLLLoss into a single class.

It’s commonly used for multi-class classification tasks.

It expects:
- Raw, unnormalized outputs from the model (logits).
- Integer class labels (e.g., 0, 1, 2…) as the ground truth.

Internally:
- It applies the softmax function to the logits, converting them into probabilities.
- Then it calculates the negative log-likelihood of the correct class.
- The result is a single scalar loss value — a measure of how wrong the model was on average.

This loss is then used by the optimizer during backpropagation to update the model’s weights in a direction that reduces future error.

Layman’s explanation:
- It takes the model’s raw guesses (logits) and turns them into probabilities.
- It checks how close those probabilities are to the actual answer.
- If the model is very wrong (low probability on the correct class), it gets “punished” with a high loss.
- This punishment teaches the model to adjust and improve its predictions over time.
"""

### TensorBoard Logging ###

writer = SummaryWriter() # Initialize TensorBoard writer for logging
# This allows us to visualize the training process, including loss and accuracy metrics, in TensorBoard.

### Gradient Scaling ###

scaler = GradScaler(enabled=USE_AMP) # Initialize gradient scaler for mixed precision training (if enabled in the configuration)
# This is useful for reducing memory usage and speeding up training on GPUs that support it.


# /-----------------------------------------------------------------------------------------------------------------------------------/

### Training loop ###
# The training loop is where the model learns from the data.

## Loop Per Epoch

def train_epoch(model, data_loader, loss_fn, optimizer, device, scaler, use_amp, accumulation_steps):
    """
    Defines how the model is trained for each epoch.
    """
    model.train() # Set the model to training mode
    running_loss = 0.0 # Initialize the running loss to zero
    last_loss = 0.0 # Initialize the last loss to zero        
    optimizer.zero_grad() # Zero the gradients at the start of the epoch

    for i, batch in enumerate(tqdm(data_loader, desc="Training")):
        # Move the batch to the specified device (GPU or CPU)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass (compute the model outputs)
        with autocast(enabled=use_amp, device_type='cuda' if USE_CUDA else 'cpu'): # This enables automatic mixed precision training (if enabled in the configuration)
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels) # Compute the loss

        # Backward pass (compute gradients)
        if use_amp:
            scaler.scale(loss / accumulation_steps).backward()
        else:
            (loss / accumulation_steps).backward()

        running_loss += loss.item()
        last_loss = loss.item()

        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:

            # Gradient clipping to prevent exploding gradients
            if MAX_GRAD_NORM > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            # Update the model parameters / weights (see explanation of CrossEntropyLoss above)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad() # Zero the gradients after updating the model parameters

        # Print the loss every 100 steps
        if (i + 1) % 100 == 0:
            print(f"Step {i + 1}/{len(data_loader)}, Loss: {last_loss:.4f}")
        
    return running_loss / len(data_loader) # Return the average loss for the epoch

def eval_model(model, data_loader, loss_fn, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval() # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0 # Initialize the number of correct predictions to zero
    total = 0 # Initialize the total number of predictions to zero

    with torch.no_grad(): # Disable gradient calculation for evaluation
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass (compute the model outputs)
            outputs = model(input_ids, attention_mask)

            # Compute the loss
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            # Get predictions and calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total # Calculate the accuracy
    return val_loss / len(data_loader), accuracy # Return the average loss and accuracy

def train():
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        # Train the model for one epoch
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scaler, USE_AMP, ACCUMULATION_STEPS)

        # Evaluate the model on the validation set
        val_loss, val_accuracy = eval_model(model, val_loader, loss_fn, device)

        # Log the losses to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        # Print the training and validation loss and accuracy (good for debugging)
        # This is useful for monitoring the training process and ensuring that the model is learning.
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save the model checkpoint
        torch.save(model.state_dict(), f"{CHECKPOINTS_PATH}/{NICKNAME}_epoch_{epoch + 1}.pth")
        print(f"Model checkpoint saved for epoch {epoch + 1}")

if __name__ == "__main__":
    train() # Start the training process
    writer.close() # Close the TensorBoard writer after training is complete
    print("Training complete!")
    # This ensures that all resources are released and the logs are saved properly.
    # It’s a good practice to close the writer to avoid any potential memory leaks or file corruption.
    print("TensorBoard logs saved!")
    print("Model checkpoints saved at training/checkpoints/")
    print("Model training complete! Good job!!!")

"""
### Summary ###
- The training script orchestrates the entire training process, including data loading, model instantiation, loss computation, backpropagation, and checkpoint saving.
- It uses the DistilBERT architecture for sequence classification tasks, making it suitable for our task (political bias classification).
- The script is designed to be modular and easy to understand, with clear separation of concerns for each component (data loading, model definition, training loop, etc.).
- The training process is logged using TensorBoard, allowing for easy monitoring of the training progress and performance metrics.
- The model checkpoints are saved after each epoch, allowing for easy recovery and further training if needed.
- The script is designed to be run as a standalone program, with the main function orchestrating the training process.
- The script is well-commented, providing explanations for each step and making it easy to understand the flow of the training process.
- The script is designed to be easily extensible, allowing for future modifications and improvements as needed.
- The script is designed to be run on a GPU if available, providing faster training times and improved performance.
- The script uses AMP (Automatic Mixed Precision) for faster training and reduced memory usage, if enabled in the configuration.
- The script uses gradient accumulation to simulate larger batch sizes, allowing for more efficient training on smaller GPUs.
"""