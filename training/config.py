from datetime import datetime as dt
import os

### Model Configuration ###
MODEL_NAME         = "distilbert-base-uncased"
NUM_LABELS         = 3

### Training Configuration ###
NUM_EPOCHS         = 8              # Fewer epochs may suffice for 14k rows
BATCH_SIZE         = 16             # Reduce if running into memory issues
EVAL_BATCH_SIZE    = 16
LEARNING_RATE      = 2e-5
WEIGHT_DECAY       = 0.05           # Increase weight decay for stronger regularization
WARMUP_STEPS       = 200            # Lower, since total steps will be fewer
MAX_GRAD_NORM      = 1.0
ACCUMULATION_STEPS = 5              # Set to 1 unless you need gradient accumulation

### Optimizer (AdamW) ###
OPTIMIZER_EPS      = 1e-8

### DataLoader Configuration ###
NUM_WORKERS        = 6              # Fewer workers may be sufficient
PIN_MEMORY         = True
SHUFFLE_DATA       = True
TRAIN_SPLIT        = 0.8
AUGMENT            = True           # Enable data augmentation
AUG_PERCENTAGE     = 0.3            # Increase augmentation percentage

### Sequence Configuration ###
MAX_SEQ_LENGTH     = 254 # Maximum length for DistilBERT
TRUNCATION         = True
PAD_TO_MAX_LENGTH  = True
ADD_SPECIAL_TOKENS = True

### GPU Configuration ###
USE_CUDA           = True
CUDA_DEVICE        = 0

### AMP (Mixed Precision) ###
USE_AMP            = True

### Miscellaneous ###
SEED               = 42
DATASET_PATH       = "./data/datasets/allsides_data_unstructured.csv"
NICKNAME           = f"{NUM_EPOCHS}_epochs_{LEARNING_RATE}_lr_{MAX_SEQ_LENGTH}_seqlen"
CHECKPOINTS_PATH   = os.path.join("trained_models", NICKNAME, "checkpoint")
LOG_PATH           = os.path.join("logs", NICKNAME, dt.now().strftime("%Y-%m-%d_%H-%M-%S"))