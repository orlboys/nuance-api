import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

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
        """
        super(BiasModel, self).__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS
        )
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_ids, attention_mask):
        """
        Defines how the model should pass the input through the network.
        """
        outputs = self.model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:,0]  # CLS token
        dropped = self.dropout(pooled_output)
        logits = self.model.classifier(dropped)
        return logits

# ---------------- loading Tokenizer ----------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # change the model name if needed

# ---------------- loading Model ----------------
model = BiasModel()
model.load_state_dict(torch.load(".././models/model_15_06_25.pth", map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# ---------------- Function to Analyze Bias ----------------
async def analyze(text: str) -> dict:
    """
    Analyzes the bias of a given text using the pre-trained BiasModel.
    :param text: The text to be analyzed.
    :return: A dictionary containing the bias prediction and probabilities.
    """
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    
    with torch.no_grad():
        logits = model(tokens['input_ids'], tokens['attention_mask'])
        probs = torch.softmax(logits, dim=1).squeeze()
        prediction = torch.argmax(probs).item()

    return {
        "prediction": prediction,
        "probabilities": {
            "left": probs[0].item(),
            "right": probs[1].item(),
            "compound": probs[2].item()
        }
    }