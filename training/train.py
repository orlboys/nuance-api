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
    LOG_PATH
)

import os
import sys

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np
import datetime as dt

def main():
    # Check if the script is being run from the correct directory
    if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
        print("⚠️: This script must be run from the training directory.")
        print("Please navigate to the training directory and run the script again.")
        sys.exit(1)

    if not os.path.exists(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH) # Create the checkpoints directory if it doesn't exist
        print(f"✅: Checkpoint path {CHECKPOINTS_PATH} created.")

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH) # Create the log directory if it doesn't exist
        print(f"✅: Log path {LOG_PATH} created.")

    ### GPU Configuration ###
    device = torch.device("cuda") # Check if CUDA is available and set the device accordingly

    ### Data Loading ###

    print(f"\n🔍 Loading data from: {DATASET_PATH}")
    train_loader, val_loader = create_dataloaders(DATASET_PATH)
    print(f"✅ Data loaded successfully:")
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")

    ### Model, Loss, Optimizer ###

    print(f"\n🤖 Initializing model on device: {device}")
    model = BiasModel().to(device)
    print(f"✅ Model architecture:")
    print(model)

    #### Compute Class Weights
    print("\n⚖️ Computing class weights for imbalanced dataset...")
    # Get the Subset object
    subset = train_loader.dataset

    # Get the original dataset and the actual indices used for training
    full_dataset = subset.dataset
    train_indices = subset.indices

    # Extract the labels using the correct indices
    y_train = [full_dataset.labels[i] for i in train_indices]

    # Compute class weights
    classes, counts = np.unique(y_train, return_counts=True)
    n_samples = len(y_train)
    n_classes = len(classes)

    class_weights = {cls: n_samples / (n_classes * count) for cls, count in zip(classes, counts)}
    print("📊 Class distribution:")
    for cls, count in zip(classes, counts):
        print(f"   Class {cls}: {count} samples, weight: {class_weights[cls]:.4f}")

    class_weights_tensor = torch.tensor([class_weights[cls] for cls in range(n_classes)], dtype=torch.float).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor) # Loss function for multi-class classification (CrossEntropyLoss is suitable for multi-class classification tasks)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        eps=OPTIMIZER_EPS,
        weight_decay=WEIGHT_DECAY
    )

    # Add ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # Reduce LR by half
        patience=1,  # Wait 1 epoch of no improvement
        verbose=True,
        min_lr=1e-7
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

    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format with time separators
    run_name = f"{NICKNAME}_{timestamp}"
    log_dir = os.path.join(LOG_PATH, run_name)

    # Initialize TensorBoard writer with additional metadata
    writer = SummaryWriter(log_dir=log_dir)

    # Log hyperparameters
    writer.add_hparams(
        {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'weight_decay': WEIGHT_DECAY,
            'max_grad_norm': MAX_GRAD_NORM,
            'optimizer_eps': OPTIMIZER_EPS,
            'num_epochs': NUM_EPOCHS,
            'model_name': MODEL_NAME,
            'use_amp': USE_AMP,
            'accumulation_steps': ACCUMULATION_STEPS,
        },
        {'dummy': 0}  # Required metric dict, not used
    )

    # Add model graph to TensorBoard
    dummy_input_ids = torch.zeros((1, 512), dtype=torch.long).to(device)
    dummy_attention_mask = torch.ones((1, 512), dtype=torch.long).to(device)
    writer.add_graph(model, (dummy_input_ids, dummy_attention_mask))

    print(f"✅ TensorBoard logging initialized at: {log_dir}")

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
        if len(data_loader) == 0:
            print("⚠️: Data loader is empty. Skipping training for this epoch.")
            return 0.0, 0.0  # Return default loss and accuracy values    model.train() # Set the model to training mode
        running_loss = 0.0 # Initialize the running loss to zero
        last_loss = 0.0 # Initialize the last loss to zero
        correct = 0 # Initialize the number of correct predictions to zero
        total = 0 # Initialize the total number of predictions to zero

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

            # Calculate predictions and update accuracy metrics
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:

                # Gradient clipping to prevent exploding gradients
                if MAX_GRAD_NORM > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)            # Update the model parameters / weights (see explanation of CrossEntropyLoss above)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                optimizer.zero_grad() # Zero the gradients before the next batch

            # Debug batch information every 100 steps
            if (i + 1) % 100 == 0:
                print(f"\n📦 Batch {i + 1} stats:")
                print(f"   - Input shape: {input_ids.shape}")
                print(f"   - Current loss: {last_loss:.4f}")
                print(f"   - Accuracy so far: {(correct/total)*100:.2f}%")
                print(f"   - Memory used: {torch.cuda.memory_allocated()/1024**2:.1f}MB") if USE_CUDA else None

        return running_loss / len(data_loader), correct / total # Return the average loss and accuracy for the epoch

    def eval_model(model, data_loader, loss_fn, device, return_preds=False):
        """
        Evaluates the model on the validation set.
        If return_preds is True, also returns all true and predicted labels for classification report.
        """
        if len(data_loader) == 0:
            print("⚠️: Validation data loader is empty. Skipping evaluation.")
            return float('inf'), 0.0 if not return_preds else (float('inf'), 0.0, [], [])

        model.eval() # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0 # Initialize the number of correct predictions to zero
        total = 0 # Initialize the total number of predictions to zero
        all_labels = []
        all_preds = []

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

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        accuracy = correct / total # Calculate the accuracy
        if return_preds:
            return val_loss / len(data_loader), accuracy, all_labels, all_preds
        return val_loss / len(data_loader), accuracy # Return the average loss and accuracy

    def train():
        print("\n🚀 Starting training process...")
        print(f"   - Epochs: {NUM_EPOCHS}")
        print(f"   - Learning rate: {LEARNING_RATE}")
        print(f"   - Batch size: {BATCH_SIZE}")
        print(f"   - Device: {device}")
        print(f"   - AMP enabled: {USE_AMP}")
        
        patience = 3  # Number of epochs to wait for improvement
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(NUM_EPOCHS):
            print(f"\n{'='*50}")
            print(f"⭐ Epoch {epoch + 1}/{NUM_EPOCHS}")
            print(f"{'='*50}")

            # Train the model for one epoch
            train_loss, train_accuracy = train_epoch(model, train_loader, loss_fn, optimizer, device, scaler, USE_AMP, ACCUMULATION_STEPS)

            # Evaluate the model on the validation set and get predictions
            val_loss, val_accuracy, val_true, val_pred = eval_model(model, val_loader, loss_fn, device, return_preds=True)

            # Log the losses to TensorBoard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_accuracy, epoch)
            writer.add_scalar("Accuracy/train", train_accuracy, epoch)
            # Log the learning rate to TensorBoard
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
            # Log the model parameters to TensorBoard (optional, can be removed if not needed)
            if epoch % 2 == 0: # Log model parameters every 2 epochs
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f"{name}.grad", param.grad, epoch)

            # Print the training and validation loss and accuracy (good for debugging)
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            # Step the scheduler with the validation loss
            scheduler.step(val_loss)
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

            # Enhanced progress reporting
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"   - Training Loss: {train_loss:.4f}")
            print(f"   - Validation Loss: {val_loss:.4f}")
            print(f"   - Training Accuracy: {train_accuracy*100:.2f}%")
            print(f"   - Validation Accuracy: {val_accuracy*100:.2f}%")
            if USE_CUDA:
                print(f"   - GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                print(f"   - GPU Utilization: {torch.cuda.utilization()}%")

            # Print classification report for validation set
            print("\nClassification Report (Validation):")
            print(classification_report(val_true, val_pred))

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {patience} epochs with no improvement.")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break

            # Optionally, save checkpoints at specific intervals (e.g., every 2 epochs)
            if (epoch + 1) % 2 == 0:
                checkpoint_path = f"{CHECKPOINTS_PATH}/{NICKNAME}_epoch_{epoch + 1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(), # We save the model state dict, which contains the model parameters. This is more efficient than saving the entire model, but can still be loaded later to recreate the model.
                    'optimizer_state_dict': optimizer.state_dict(), # We save the optimizer state dict, which contains the optimizer parameters. This is useful for resuming training later.
                    'scaler_state_dict': scaler.state_dict() if USE_AMP else None, # We save the scaler state dict, which contains the scaler parameters. This is useful for resuming training later with mixed precision.
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    }, checkpoint_path)
                print(f"Model checkpoint saved for epoch {epoch + 1}")

    print("\n🔧 Initializing training environment...")
    train()
    writer.close()
    print("\n✅ Training complete!")
    print("📊 TensorBoard logs saved!")
    print(f"💾 Model checkpoints saved at: {CHECKPOINTS_PATH}")
    print("\n🎉 Training process finished successfully! Good job!!!")

if __name__ == "__main__":
    main()

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