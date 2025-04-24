#  Implements a custom Dataset class that loads and preprocesses your data, including tokenization and encoding.
#  This is a PyTorch Dataset class that loads and preprocesses the data for training a model. It includes tokenization and encoding of the text data using a pre-trained BERT tokenizer.

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer # Pre-trained BERT tokenizer rather than a custom one - this is a common practice in NLP tasks.
from config import (
    MODEL_NAME, MAX_SEQ_LENGTH, TRUNCATION, 
    PAD_TO_MAX_LENGTH, ADD_SPECIAL_TOKENS
)

"""
Since we're using a pre-trained DistilBERT model, we need to tokenize our text data in a way that DistilBERT can understand.
Luckily, the Hugging Face Transformers library provides a convenient way to do this with the DistilBertTokenizer.
The tokenizer will handle the following:
- Preprocessing the text (lowercasing, removing special characters, etc.)
- Tokenizing the text (breaking it down into subwords or tokens)
- Adding special tokens (like [CLS] and [SEP])
- Padding and truncating the sequences to a fixed length (if necessary)
- Creating attention masks (to indicate which tokens are padding and which are not)
etc.
"""

"""
NOTE: With BERT models, and specifically distilbert-base-lowercase, we DO NOT
- Remove stop words (BERT can handle them)
- Use stemming or lemmatization (BERT uses subword tokenization)
- Strip punctuation (BERT can handle it)
"""

class BiasDataset(Dataset):
    """
    Custom Dataset class for loading
    and preprocessing the bias detection dataset.
    """
    def __init__(self, csv_file):
        """
        Initializes the dataset with a CSV file, tokenizer, and maximum sequence length.

        Args:
            csv_file (str): Path to the CSV file containing the dataset.
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        self.texts = self.data['text'].tolist() # List of texts
        self.labels = self.data['label'].tolist() # List of labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves a single data point from the dataset.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            dict: A dictionary containing the tokenized input IDs, attention masks, and label.
        """
        text = self.texts[idx] # Get the text at the given index
        label = self.labels[idx] # Get the label at the given index
        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=ADD_SPECIAL_TOKENS, # Add [CLS] and [SEP] tokens - these are standard in BERT
            max_length=MAX_SEQ_LENGTH,
            padding='max_length' if PAD_TO_MAX_LENGTH else 'do_not_pad', # Pad to max_length (common practice in NLP tasks)
            truncation=TRUNCATION, # Truncate if longer than max_length
            return_tensors='pt', # Return PyTorch tensors
            return_attention_mask=True,
        )

        return {
            'input_ids': encoded_text['input_ids'].squeeze(0), # Remove the batch dimension
            'attention_mask': encoded_text['attention_mask'].squeeze(0), # Remove the batch dimension
            'label': torch.tensor(label, dtype=torch.long) # Convert label to tensor
        }

        """
        encoded_text is in the form of a dictionary with keys 'input_ids' and 'attention_mask'.
        - 'input_ids' are the token IDs for the text, including special tokens like [CLS] and [SEP].
        - 'attention_mask' indicates which tokens are padding (0) and which are not (1).
        Example:
        {
            'input_ids': tensor([[ 101,  2054,  2003,  ... , 0, 0, 0]]),
            'attention_mask': tensor([[1, 1, 1, ... , 0, 0, 0]])
        }
        Note that 101 is [CLS], 102 is [SEP], and 0 is the padding token.

        Input id is what BERT sees as 'input'
        Attention mask is what BERT sees as 'mask'
            - It tells BERT which tokens to pay attention to and which ones to ignore (the padding ones).
        """