import os 
import torch 


# Paths 
PATH_LOGS = os.environ.get("PATH_LOGS", "./runs")

PATH_BASE_MODELS = os.environ.get("PATH_BASE_MODELS", "./base_models")
PATH_CACHE_DATASETS = os.environ.get("PATH_CACHE_DATASETS", "./data/cache")

PATH_CHECKPOINT_MODELS = os.environ.get("PATH_CHECKPOINT_MODELS", "./models")

PATH_SAVE_NL_ENCODER = os.environ.get("PATH_SAVE_NL_ENCODER", "./models/codecg-nl-encoder")
PATH_SAVE_NL_DECODER = os.environ.get("PATH_SAVE_NL_DECODER", "./models/codecg-nl-decoder")

# Hyperparams 
GLOBAL_SEED = 42 

TOKENIZER_MODEL = "Salesforce/codet5-base"
NL_ENCODER_BASE_MODEL = "roberta-base"
NL_DECODER_BASE_MODEL = "gpt2"
VOCAB_SIZE = 32100

MAX_SEQUENCE_LENGTH = 128
PADDING = "max_length"

MAX_EPOCHS = 50
BATCH_SIZE = 32

LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 1e-3

DATASET_NAME = "code_x_glue_ct_code_to_text"
PL = "python"

# Hardware
NUM_WORKERS = min(4, int(os.cpu_count() / 2))
AVAIL_GPUS = min(1, torch.cuda.device_count())

# Project
PROJECT_NAME = "CodeCG"