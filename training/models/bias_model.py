# This file defines the model architecture and instantiates the model for training.
# It uses the DistilBERT architecture for sequence classification tasks
# This makes it suitable for our task (political bias classification).

from torch import nn
from transformers import DistilBertForSequenceClassification
from config import MODEL_NAME, NUM_LABELS

# This is a PyTorch model that uses the DistilBERT architecture for sequence classification tasks.
# We define it as class BiasModel, which inherits from nn.Module.
# This allows us to create a custom model that can be trained and evaluated using PyTorch.

class BiasModel(nn.Module):
    
    def __init__(self, dropout_prob=0.3):
        """
        Initializes the BiasModel with a pre-trained DistilBERT model for sequence classification.
        Adds a dropout layer for regularization.
        """
        super(BiasModel, self).__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS
        )
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model with dropout regularization.
        Args:
            input_ids (torch.Tensor): Input IDs for the DistilBERT model.
            attention_mask (torch.Tensor): Attention mask for the DistilBERT model.
        Returns:
            torch.Tensor: Output logits from the model.
        """
        outputs = self.model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:,0]  # CLS token
        dropped = self.dropout(pooled_output)
        logits = self.model.classifier(dropped)
        return logits # Output logits for classification - these are the raw scores for each class.
        # e.g. nn.CrossEntropyLoss() expects raw logits as input, not probabilities, to compute the loss.
        # e.g. using a softmax to convert logits to probabilites.