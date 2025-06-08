# =============================================================================
# CHECKPOINT-BASED MODEL DEBUGGING SCRIPT WITH SEPARATE TEST SET
# =============================================================================

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import glob
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# CONFIGURATION - MODIFY THESE TO MATCH YOUR SETUP
# =============================================================================

CHECKPOINTS_PATH = "./trained_models/8_epochs_2e-05_lr_512_seqlen/checkpoint/"
# SEPARATE PATHS FOR TRAINING AND TEST DATA
TRAIN_DATASET_PATH = "data/datasets/allsides_data_unstructured.csv"  # Original training data
TEST_DATASET_PATH = "data/datasets/test.csv" # Separate test set for evaluation
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3
BATCH_SIZE = 16
MAX_LENGTH = 128

# =============================================================================
# MODEL AND DATASET CLASSES
# =============================================================================

class BiasModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_labels=NUM_LABELS, dropout_prob=0.3):
        super(BiasModel, self).__init__()
        from transformers import DistilBertForSequenceClassification
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:,0]
        dropped = self.dropout(pooled_output)
        logits = self.model.classifier(dropped)
        return logits

class BiasDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=MAX_LENGTH):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate required columns
        required_columns = ['text', 'label']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {csv_file}: {missing_columns}")
        
        self.texts = self.data['text'].tolist()
        self.labels = self.data['label'].tolist()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# =============================================================================
# DATASET LOADING WITH TEST SET OPTION
# =============================================================================

def create_dataloader(dataset_path, batch_size=BATCH_SIZE, dataset_type="test"):
    """
    Create DataLoader for either training or test dataset
    
    Args:
        dataset_path: Path to CSV file
        batch_size: Batch size for DataLoader
        dataset_type: "train" or "test" for logging purposes
    """
    if not os.path.exists(dataset_path):
        print(f"❌ {dataset_type.title()} dataset file '{dataset_path}' not found!")
        return None, None
    
    print(f"📊 Loading {dataset_type} dataset from: {dataset_path}")
    
    try:
        # Initialize tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        
        # Create dataset
        dataset = BiasDataset(dataset_path, tokenizer)
        
        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        print(f"   ✅ {dataset_type.title()} dataset loaded: {len(dataset)} samples")
        print(f"   ✅ DataLoader created: {len(dataloader)} batches")
        
        # Show label distribution
        labels = dataset.labels
        label_counts = Counter(labels)
        print(f"   📈 Label distribution: {dict(sorted(label_counts.items()))}")
        
        return dataloader, dataset
        
    except Exception as e:
        print(f"   ❌ Error loading {dataset_type} dataset: {str(e)}")
        return None, None

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def run_checkpoint_diagnosis(checkpoint_path=None, test_dataset_path=TEST_DATASET_PATH, 
                           train_dataset_path=None):
    """
    Run complete diagnosis on a checkpoint using separate test set
    
    Args:
        checkpoint_path: Path to checkpoint file (None for most recent)
        test_dataset_path: Path to test CSV file
        train_dataset_path: Optional path to training CSV for comparison
    """
    print("🚀 STARTING CHECKPOINT-BASED DIAGNOSIS WITH TEST SET")
    print("="*70)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Find checkpoints if not specified
    if checkpoint_path is None:
        checkpoints = find_available_checkpoints()
        if not checkpoints:
            return None
        checkpoint_path = checkpoints[0]
        print(f"\n📦 Using most recent checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Load model
    model = load_model_from_checkpoint(checkpoint_path, device)
    if model is None:
        return None
    
    # Load TEST dataset
    test_dataloader, test_dataset = create_dataloader(test_dataset_path, dataset_type="test")
    if test_dataloader is None:
        return None
    
    # Run analysis on TEST set
    print(f"\n🎯 EVALUATING ON TEST SET")
    test_results = analyze_checkpoint_predictions(
        model, test_dataloader, device, 
        f"{os.path.basename(checkpoint_path)} (TEST SET)"
    )
    
    # Optional: Compare with training set performance
    if train_dataset_path and os.path.exists(train_dataset_path):
        print(f"\n📊 COMPARING WITH TRAINING SET PERFORMANCE")
        print("="*60)
        
        train_dataloader, train_dataset = create_dataloader(train_dataset_path, dataset_type="train")
        if train_dataloader is not None:
            train_results = analyze_checkpoint_predictions(
                model, train_dataloader, device, 
                f"{os.path.basename(checkpoint_path)} (TRAINING SET)"
            )
            
            # Compare results
            compare_train_test_performance(train_results, test_results)
    
    # Diagnose issues
    diagnose_model_issues(test_results)
    
    return test_results

def compare_train_test_performance(train_results, test_results):
    """
    Compare performance between training and test sets to detect overfitting
    """
    print("\n🔄 TRAIN vs TEST COMPARISON:")
    print("-" * 50)
    
    train_acc = train_results.get('accuracy', 0)
    test_acc = test_results.get('accuracy', 0)
    
    print(f"   Training Accuracy: {train_acc:.4f} ({train_acc*100:.1f}%)")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"   Accuracy Drop: {(train_acc - test_acc):.4f} ({(train_acc - test_acc)*100:.1f}%)")
    
    # Overfitting detection
    if train_acc - test_acc > 0.1:
        print("   🚨 POTENTIAL OVERFITTING DETECTED!")
        print("   💡 Suggestions:")
        print("      - Increase regularization (dropout)")
        print("      - Use early stopping")
        print("      - Collect more training data")
        print("      - Reduce model complexity")
    elif train_acc - test_acc < 0.05:
        print("   ✅ Good generalization - no significant overfitting")
    else:
        print("   ⚠️  Moderate performance gap - monitor closely")

def compare_multiple_checkpoints_on_test(checkpoint_dir=CHECKPOINTS_PATH, 
                                       test_dataset_path=TEST_DATASET_PATH):
    """
    Compare multiple checkpoints using the test set
    """
    print("🔄 COMPARING MULTIPLE CHECKPOINTS ON TEST SET")
    print("="*70)
    
    checkpoints = find_available_checkpoints(checkpoint_dir)
    if len(checkpoints) < 2:
        print("❌ Need at least 2 checkpoints for comparison")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataloader, _ = create_dataloader(test_dataset_path, dataset_type="test")
    
    if test_dataloader is None:
        return
    
    comparison_results = []
    
    for checkpoint_path in checkpoints[:5]:  # Compare up to 5 most recent
        print(f"\n📦 Analyzing: {os.path.basename(checkpoint_path)}")
        
        model = load_model_from_checkpoint(checkpoint_path, device)
        if model is None:
            continue
        
        # Quick analysis on test set
        model.eval()
        predictions = []
        labels = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask)
                batch_preds = torch.argmax(logits, dim=1)
                
                predictions.extend(batch_preds.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
                
                correct += (batch_preds == batch_labels).sum().item()
                total += batch_labels.size(0)
        
        accuracy = correct / total
        pred_counts = Counter(predictions)
        unique_preds = len(set(predictions))
        
        comparison_results.append({
            'checkpoint': os.path.basename(checkpoint_path),
            'test_accuracy': accuracy,
            'unique_predictions': unique_preds,
            'prediction_distribution': dict(pred_counts)
        })
    
    # Display comparison
    print("\n📊 CHECKPOINT COMPARISON ON TEST SET:")
    print("-" * 60)
    
    for result in comparison_results:
        print(f"📦 {result['checkpoint']}:")
        print(f"   Test Accuracy: {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.1f}%)")
        print(f"   Unique predictions: {result['unique_predictions']}")
        print(f"   Distribution: {result['prediction_distribution']}")
        print()

def quick_test_on_test_set(checkpoint_path, test_dataset_path=TEST_DATASET_PATH, num_samples=100):
    """
    Quick validation on test set
    """
    print(f"⚡ QUICK TEST ON TEST SET: {os.path.basename(checkpoint_path)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(checkpoint_path, device)
    
    if model is None:
        return False
    
    test_dataloader, _ = create_dataloader(test_dataset_path, dataset_type="test")
    if test_dataloader is None:
        return False
    
    predictions = []
    labels = []
    samples_processed = 0
    
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            if samples_processed >= num_samples:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels']
            
            logits = model(input_ids, attention_mask)
            batch_preds = torch.argmax(logits, dim=1)
            
            predictions.extend(batch_preds.cpu().numpy())
            labels.extend(batch_labels.numpy())
            
            samples_processed += len(batch_preds)
    
    accuracy = np.mean(np.array(predictions) == np.array(labels))
    pred_counts = Counter(predictions)
    unique_preds = len(set(predictions))
    
    print(f"   Samples tested: {len(predictions)}")
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"   Unique predictions: {unique_preds}")
    print(f"   Distribution: {dict(pred_counts)}")
    
    if unique_preds == 1:
        print("   ❌ Still stuck on one class!")
        return False
    else:
        print("   ✅ Predicting multiple classes on test set!")
        return True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_available_checkpoints(checkpoint_dir=CHECKPOINTS_PATH):
    """Find all available checkpoint files"""
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory '{checkpoint_dir}' not found!")
        return []
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in '{checkpoint_dir}'")
        return []
    
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"✅ Found {len(checkpoint_files)} checkpoint files:")
    for i, file in enumerate(checkpoint_files):
        filename = os.path.basename(file)
        mod_time = os.path.getmtime(file)
        import datetime
        date_str = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"   {i+1}. {filename} (Modified: {date_str})")
    
    return checkpoint_files

def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """Load model from checkpoint file"""
    print(f"📦 Loading model from: {os.path.basename(checkpoint_path)}")
    
    try:
        model = BiasModel()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("   ✅ Loaded from full checkpoint (with optimizer state)")
        else:
            model.load_state_dict(checkpoint)
            print("   ✅ Loaded from state dict checkpoint")
        
        model.to(device)
        model.eval()
        print(f"   📊 Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"   ❌ Error loading checkpoint: {str(e)}")
        return None

# [Include other existing functions like analyze_checkpoint_predictions, diagnose_model_issues, etc.]

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("🔧 CHECKPOINT DIAGNOSIS TOOL WITH TEST SET")
    print("="*50)
    
    # Update these paths
    CHECKPOINTS_PATH = "./trained_models/8_epochs_2e-05_lr_512_seqlen/checkpoint/"
    TEST_DATASET_PATH = "data/datasets/test.csv"  # Your test set
    TRAIN_DATASET_PATH = "data/datasets/allsides_data_unstructured.csv"  # Your training set
    
    print("Available commands:")
    print("1. Evaluate latest checkpoint on test set")
    print("2. Compare multiple checkpoints on test set")
    print("3. Quick test on test set")
    print("4. Compare train vs test performance")
    
    # Run diagnosis on test set
    results = run_checkpoint_diagnosis(
        test_dataset_path=TEST_DATASET_PATH,
        train_dataset_path=TRAIN_DATASET_PATH  # Optional for comparison
    )
    
    # Compare checkpoints on test set
    compare_multiple_checkpoints_on_test(test_dataset_path=TEST_DATASET_PATH)
    
    # Quick test
    checkpoint_path = "./trained_models/8_epochs_2e-05_lr_512_seqlen/checkpoint/8_epochs_2e-05_lr_512_seqlen_epoch_4.pth"
    quick_test_on_test_set(checkpoint_path, TEST_DATASET_PATH)