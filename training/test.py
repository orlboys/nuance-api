import torch
import pandas as pd
from torch.utils.data import DataLoader
from data.dataset import BiasDataset
from models.bias_model import BiasModel

# Replace with your actual model and dataset paths
MODEL_PATH = "./trained_models/8_epochs_2e-05_lr_512_seqlen/checkpoint/8_epochs_2e-05_lr_512_seqlen_epoch_4.pth"
TEST_DATA_PATH = "./data/datasets/test_data.csv"

def test_model():
    """Test the trained model on test data."""
    
    # Load test dataset using your existing BiasDataset class
    test_dataset = BiasDataset(TEST_DATA_PATH, augment=False)  # No augmentation for testing
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=16,  # Adjust batch size as needed
        shuffle=False
    )
    
    # Load the trained model
    model = BiasModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))  # Load model state dict
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Run inference
    all_predictions = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Get predictions
            _, predicted = torch.max(logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Show some sample predictions
    print(f"\nSample predictions (first 10):")
    for i in range(min(10, len(all_predictions))):
        print(f"Predicted: {all_predictions[i]}, Actual: {all_labels[i]}")

if __name__ == "__main__":
    test_model()