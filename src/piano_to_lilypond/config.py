import os
import torch

# Data paths
DATA_DIR = os.getenv("P2L_DATA_DIR", "./data")
MAESTRO_DIR = os.path.join(DATA_DIR, "maestro/maestro-v3.0.0")  # Expect audio + MIDI pairs
SYNTHTRAIN_DIR = os.path.join(DATA_DIR, "synthetic")  # Synthetic MIDI renders

# Audio processing
SAMPLE_RATE = 44100
N_MELS = 229
HOP_LENGTH = 512  # ~11.6ms
WIN_LENGTH = 2048  # ~46ms

# Model hyperparameters
ENC_FEAT_DIM = 64
ENC_HIDDEN_DIM = 256
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
NUM_HEADS = 4
FFN_DIM = 1024
DROPOUT = 0.1
VOCAB_SIZE = None  # to be set after tokenization

# Training
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 10000
TOTAL_STEPS = 250000
AUX_CTC_WEIGHT = 0.3

# Memory management
MAX_AUDIO_LENGTH = 20000
GRADIENT_ACCUMULATION_STEPS = 16

# Inference
BEAM_SIZE = 3
MAX_DECODING_STEPS = 10000

# LilyPond
LILYPOND_CMD = "lilypond"

# Checkpointing
CHECKPOINT_DIR = os.getenv("P2L_CKPT_DIR", "./checkpoints")

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"