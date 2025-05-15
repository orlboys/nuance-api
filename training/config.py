### Model Configuration ###
MODEL_NAME         = "distilbert-base-uncased"
NUM_LABELS         = 3

### Training Configuration ###
NUM_EPOCHS         = 15
BATCH_SIZE         = 64        # increase if GPU allows
EVAL_BATCH_SIZE    = 64
LEARNING_RATE      = 2e-5
WEIGHT_DECAY       = 1e-2
WARMUP_STEPS       = 1000      # ~5% of 20,000 steps
MAX_GRAD_NORM      = 1.0
ACCUMULATION_STEPS = 2

### Optimizer (AdamW) ###
OPTIMIZER_EPS      = 1e-8

### DataLoader Configuration ###
NUM_WORKERS        = 6
PIN_MEMORY         = True
SHUFFLE_DATA       = True
TRAIN_SPLIT        = 0.8       # 80% training, 20% validation
AUGMENT            = False
AUG_PERCENTAGE     = 0.2

### Sequence Configuration ###
MAX_SEQ_LENGTH     = 128
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
DATASET_PATH       = "./data/datasets/allsides_data_unstructured.csv"  # Update to your new dataset path
NICKNAME           = f"newdataset20k_{NUM_EPOCHS}_epochs_{LEARNING_RATE}_lr_testmodel"
CHECKPOINTS_PATH   = f"trained_models/{NICKNAME}/checkpoint"
LOG_PATH           = f"logs/{NICKNAME}"