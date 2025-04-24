#  Orchestrates the training process, including data loading, model instantiation, loss computation, backpropagation, and checkpoint saving.
# # This script is responsible for the actual TRAINING of the model
import torch
import torch.nn as nn
import torch.optim as optim # Optimizer for training
import torch.nn.functional as F # Functional module for various operations
import models.bias_model as bias_model
import data.dataset as dataset
from config import LEARNING_RATE, WEIGHT_DECAY, MAX_GRAD_NORM

loss_fn = nn.CrossEntropyLoss() # Loss function for multi-class classification (CrossEntropyLoss is suitable for multi-class classification tasks)
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

This loss is then used during backpropagation to update the model’s weights in a direction that reduces future error.

Layman’s explanation:
- It takes the model’s raw guesses (logits) and turns them into probabilities.
- It checks how close those probabilities are to the actual answer.
- If the model is very wrong (low probability on the correct class), it gets “punished” with a high loss.
- This punishment teaches the model to adjust and improve its predictions over time.
"""

# Optimizer for training
optimizer = optim.AdamW(
    bias_model.parameters(),
    lr=LEARNING_RATE, # Learning rate for the optimizer
    eps=1e-8, # Epsilon value for numerical stability
    weight_decay=WEIGHT_DECAY # Weight decay for regularization
)