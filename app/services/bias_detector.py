import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification as Dis
from training.config import (
    MODEL_NAME, MAX_SEQ_LENGTH, TRUNCATION,
    PAD_TO_MAX_LENGTH, ADD_SPECIAL_TOKENS
)

# ---------------- Defining the Model Architecture ----------------
# This file defines the model architecture and instantiates the model for training.
# It uses the DistilBERT architecture for sequence classification tasks,
# which makes it suitable for our task (political bias classification).
# look at ./training for the training script, and more on how the model was created

# redefinition of the BiasModel class

class BiasModel(nn.Module):
    
    def __init__(self, dropout_prob=0.3):
        """
        Initializes the BiasModel with a pre-trained DistilBERT model for sequence classification.
        Adds a dropout layer for regularization.
        """
        super(BiasModel, self).__init__()
        self.model = Dis.from_pretrained(MODEL_NAME, num_labels=3)
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
        return logits

# ---------------- Device Setup ----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- Tokenizer and Model Loading ----------------
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = BiasModel()
checkpoint = torch.load("models/model_15_06_25.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

# ---------------- Function to Analyze Bias ----------------
async def predict_bias(text: str) -> dict:
    """
    Analyzes the bias of a given text using the pre-trained BiasModel.
    :param text: The text to be analyzed.
    :return: A dictionary containing the bias prediction and probabilities.
    """
    try:
        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=ADD_SPECIAL_TOKENS,
            max_length=MAX_SEQ_LENGTH,
            padding='max_length' if PAD_TO_MAX_LENGTH else 'do_not_pad',
            truncation=TRUNCATION,
            return_tensors='pt',
            return_attention_mask=True,
        )
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Handle different output formats as in test.py
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            # Get prediction (single sample)
            _, predicted = torch.max(logits, 1)
            probs = torch.softmax(logits, dim=1).squeeze()
            prediction = predicted.item()

        # Debug print for comparison with test.py
        print("\n[API Debug] Text:", text)
        print("[API Debug] Tokenized input_ids:", input_ids.cpu().numpy())
        print("[API Debug] Logits:", logits.cpu().numpy())
        print("[API Debug] Probabilities:", probs.cpu().numpy())
        print("[API Debug] Predicted class:", prediction)


        # Assuming label order: 0=left, 1=neutral, 2=right
        return {
            "left": probs[0].item(),
            "neutral": probs[1].item(),
            "right": probs[2].item(),
            # Compound score is a simple linear combination of the probabilities
            "compound": (-1) * probs[0].item() + 0 * probs[1].item() + 1 * probs[2].item(), # Compound Score - used for the 'gauge' in the UI
            "confidence": float(torch.max(probs).item()), # Returns the highest confidence a model has in its chosen class.
            "prediction": prediction,
            "error": None
        }
    except Exception as e:
        print(f"[API Error] An error occurred while analyzing bias: {str(e)}")
        return {
            "left": 0.0,
            "neutral": 0.0,
            "right": 0.0,
            "compound": 0.0,
            "prediction": -1,  # Indicating an error
            "confidence": 0.0,
            "error": str(e)
        }
