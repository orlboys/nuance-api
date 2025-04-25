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
    
    def __init__(self):
        """
        Initializes the BiasModel with a pre-trained DistilBERT model for sequence classification.
        Can be expanded to include additional layers or modifications as needed.
        """
        super(BiasModel, self).__init__()
        # Load the pre-trained DistilBERT model for sequence classification
        self.model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        Args:
            input_ids (torch.Tensor): Input IDs for the DistilBERT model.
            attention_mask (torch.Tensor): Attention mask for the DistilBERT model.
        Returns:
            torch.Tensor: Output logits from the model.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits # Output logits for classification - these are the raw scores for each class.
        # e.g. nn.CrossEntropyLoss() expects raw logits as input, not probabilities, to compute the loss.
        # e.g. using a softmax to convert logits to probabilites.