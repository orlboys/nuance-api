### Model Configuration ###
MODEL_NAME         = "distilbert-base-uncased"
NUM_LABELS         = 5

### Training Configuration ###
NUM_EPOCHS         = 8         # 8 epochs for small dataset; can increase for larger datasets
BATCH_SIZE         = 24        # reduced to fit in GPU memory
EVAL_BATCH_SIZE    = 32        # no-grad, can be larger
LEARNING_RATE      = 2e-5
WEIGHT_DECAY       = 1e-2
WARMUP_STEPS       = 0         # small dataset; can increase to ~0.1*total_steps if desired
MAX_GRAD_NORM      = 1.0
ACCUMULATION_STEPS = 2         # simulate batch_size 32 if needed

### Optimizer (AdamW) ###
OPTIMIZER_EPS      = 1e-8

### DataLoader Configuration ###
NUM_WORKERS        = 6         # CPU cores minus one
PIN_MEMORY         = True      # recommended for CUDA
SHUFFLE_DATA       = True
TRAIN_SPLIT        = 0.8       # 80% training, 20% validation

### Sequence Configuration ###
MAX_SEQ_LENGTH     = 128
TRUNCATION         = True
PAD_TO_MAX_LENGTH  = True
ADD_SPECIAL_TOKENS = True

### GPU Configuration ###
USE_CUDA           = True
CUDA_DEVICE        = 0

### AMP (Mixed Precision) ###
USE_AMP            = True      # enable torch.cuda.amp for FP16 (enabled here because my setup isn't powerful enough to not use it)

### Miscellaneous ###
SEED               = 42
DATASET_PATH       = "data/datasets/political_bias.csv"  # Path to the training dataset
NICKNAME           = "8_epoch_model(test)"  # Name of the model for saving/loading checkpoints
CHECKPOINT_PATH    = f"trained_models/{NICKNAME}/checkpoint"  # Path to save the model checkpoints