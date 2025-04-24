### Model Configuration ###
MODEL_NAME = "distilbert-base-uncased" 
NUM_LABELS = 5

### Training Configuration ###
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 0
MAX_GRAD_NORM = 1.0
NUM_WORKERS = 5 # Number of subprocesses to use for data loading (set to 0 for no multiprocessing)
                # - A general rule of thumb is to set num_workers to the number of CPU cores available MINUS 1.
PIN_MEMORY = True # Pin memory for faster data transfer to GPU (recommended for CUDA-enabled GPUs)
OPTIMIZER_EPS = 1e-8 # Epsilon value for numerical stability in the optimizer - prevents division by zero in AdamW optimizer

### Data Configuration ###
MAX_SEQ_LENGTH = 128 # Maximum sequence length for BERT
                     # - 128 is a common choice for BERT models, balancing performance and memory usage.
TRAIN_SPLIT = 0.8    # Fraction of data to use for training (0.8 = 80% for training, 20% for validation)
SHUFFLE_DATA = True  # Whether to shuffle the dataset at every epoch (recommended for better generalization)
                     # - Shuffling helps the model learn better by exposing it to different data orders during training.
TRUNCATION = True    # Whether to truncate sequences longer than MAX_SEQ_LENGTH (recommended for BERT models)
                     # - BERT can handle sequences up to 512 tokens, but shorter sequences are often used for efficiency.
PAD_TO_MAX_LENGTH = True # Whether to pad sequences to MAX_SEQ_LENGTH (recommended for BERT models)
ADD_SPECIAL_TOKENS = True  # Whether to add [CLS] and [SEP] tokens (required for BERT)

### Evaulation Configuration ###
EVAL_BATCH_SIZE = 64 # Batch size for evaluation

# NOTE: Add in breakpoints / thresholds for far right, right, centre, left, far left HERE eventually

### GPU Configuration ###
# (If you're using a GPU for training)

CUDA_DEVICE = 0 # GPU device ID to use
USE_CUDA = True # Whether to use CUDA if available

### Miscellaneous Configuration ###
SEED = 42 # Random seed for reproducibility (42 is a common choice in examples, but can be any integer)
# - Setting a seed ensures that the random processes (like weight initialization, data shuffling) produce the same results each time the code is run.