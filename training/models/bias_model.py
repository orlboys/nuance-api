# This file defines the model architecture and instantiates the model for training.
# It uses the DistilBERT architecture for sequence classification tasks
# This makes it suitable for our task (political bias classification).
# # It also includes the necessary imports and functions to load the model.

from torch import nn
from transformers import DistilBertModelforSequenceClassification

class BiasModel(nn.Module):
    def __init__(self, num_labels=5):
        """
        Initializes the BiasModel with a pre-trained DistilBERT model for sequence classification.
        Can be expanded to include additional layers or modifications as needed.
        Args:
            num_labels (int): Number of output labels for classification. Default is 5 (for 5 values of poltiical bias).
        """
        super(BiasModel, self).__init__()
        # Load the pre-trained DistilBERT model for sequence classification
        self.model = DistilBertModel.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=num_labels,
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