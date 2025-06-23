#  Implements a custom Dataset class that loads and preprocesses your data, including tokenization and encoding.
#  This is a PyTorch Dataset class that loads and preprocesses the data for training a model. It includes tokenization and encoding of the text data using a pre-trained BERT tokenizer.

import glob
import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer # Pre-trained BERT tokenizer rather than a custom one - this is a common practice in NLP tasks.
from config import (
    MODEL_NAME, MAX_SEQ_LENGTH, TRUNCATION, 
    PAD_TO_MAX_LENGTH, ADD_SPECIAL_TOKENS,
    AUG_PERCENTAGE
)
import nlpaug.augmenter.word as nas # NLP Augmentation library for data augmentation

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
    def __init__(self, json_files_path, augment=False): # augment is only True for the training set (see dataloader.py)
        """
        Initializes the dataset with a directory to JSON files containing content and bias labels, tokenizer, and maximum sequence length.

        Args:
            json_files_path (str): Path to directory containing JSON files or pattern to match JSON files
            augment (bool): whether to apply data augmentation
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        self.augment = augment # Data augmentation flag
        if self.augment:
            self.augmenter = nas.SynonymAug(aug_p=AUG_PERCENTAGE, aug_src='wordnet')

        self.texts, self.labels = self._load_json_data(json_files_path)

    def _load_json_data(self, json_files_path):
        """
        Load data from JSON files containing "content_original" and "bias" fields.
        
        Args:
            json_files_path (str): Path to directory or file pattern for JSON files
            
        Returns:
            tuple: (texts, labels) lists
        """

        texts = []
        labels = []

        # Handle both directory path and file pattern
        if os.path.isdir(json_files_path):
            json_pattern = os.path.join(json_files_path, "*.json") # selects all jsons in the directory
        else:
            json_pattern = json_files_path # if its just a single file, it just takes that one.

        json_files = glob.glob(json_pattern) # returns a list of all file paths that match the json pattern

        if not json_files:
            raise ValueError(f"No JSON files found matching pattersn {json_pattern}")
        
        print(f"Loading data from {len(json_files)} JSON files...")

        ## Get JSON contents for each file in the file path
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    ## Handling both a single object being found, and an array of objects being found (for futureproofing)
                    if isinstance(data, list):
                        for item in data:
                            if self._validate_item(item):
                                texts.append(item['content'])
                                labels.append(item['bias'])
                    else:
                        if self._validate_item(data):
                            texts.append(item['content'])
                            labels.append(item['bias'])

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Error processing file {file_path}: {e}")
                continue
        if not texts:
            raise ValueError("No valid data found in JSON files")
        
        print(f"Loaded {len(texts)} samples")
        print(f"Label distribution: {self._get_label_distribution(labels)}")
        
        return texts, labels
    
    def _validate_item(self, item):
        """
        Validate that an item has the required fields and valid values.
        
        Args:
            item (dict): Data item to validate
            
        Returns:
            bool: True if item is valid
        """
        if not isinstance(item, dict):
            return False
            
        if 'content' not in item or 'bias' not in item:
            return False
            
        if not isinstance(item['content'], str) or not item['content'].strip():
            return False
            
        if item['bias'] not in [0, 1, 2]:
            return False
            
        return True
    
    def _get_label_distribution(self, labels):
        """
        Get the distribution of labels for debugging/info purposes.
        
        Args:
            labels (list): List of labels
            
        Returns:
            dict: Label distribution
        """
        from collections import Counter
        counter = Counter(labels)
        return {
            'left (0)': counter.get(0, 0),
            'center (1)': counter.get(1, 0), 
            'right (2)': counter.get(2, 0)
        }


    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Retrieves a single data point from the dataset.

        Args:
            idx (int): Index of the data point to retrieve.

        Returns:
            dict: A dictionary containing the tokenized input IDs, attention masks, and label as tensors.
        """
        text = self.texts[idx] # Get the text at the given index
        label = self.labels[idx] # Get the label at the given index

        if self.augment:
            text = self.augment_data(text)
        
        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=ADD_SPECIAL_TOKENS, # Add [CLS] and [SEP] tokens - these are standard in BERT
            max_length=MAX_SEQ_LENGTH,
            padding='max_length' if PAD_TO_MAX_LENGTH else False, # Pad to max_length (common practice in NLP tasks)
            truncation=TRUNCATION, # Truncate if longer than max_length
            return_tensors='pt', # Return PyTorch tensors
            return_attention_mask=True,
        )

        return {
            'input_ids': encoded_text['input_ids'].squeeze(0), # Remove the batch dimension
            'attention_mask': encoded_text['attention_mask'].squeeze(0), # Remove the batch dimension
            'labels': torch.tensor(label, dtype=torch.long) # Convert label to tensor
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

    def augment_data(self, text):
        return self.augmenter.augment(text) # Synonym augmentation with 10% probability