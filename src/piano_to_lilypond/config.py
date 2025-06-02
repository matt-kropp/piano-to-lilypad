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

# Model hyperparameters (Optimized for A100 GPU)
ENC_FEAT_DIM = 256  # Increased for A100
ENC_HIDDEN_DIM = 1024  # Increased significantly for A100
NUM_ENCODER_LAYERS = 16  # Increased from original 12
NUM_DECODER_LAYERS = 16  # Increased from original 12
NUM_HEADS = 16  # Increased for A100 parallel processing
FFN_DIM = 4096  # Doubled from original 2048
DROPOUT = 0.1
VOCAB_SIZE = None  # to be set after tokenization

# Training (Optimized for A100)
BATCH_SIZE = 32  # Large batch size for A100
LEARNING_RATE = 2e-4  # Slightly higher for larger batches
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 5000  # Reduced since larger batches
TOTAL_STEPS = 100000  # Reduced since more efficient training
AUX_CTC_WEIGHT = 0.3

# Memory management (A100 optimized)
MAX_AUDIO_LENGTH = 300000  # Much longer sequences (~2.7 minutes of audio)
GRADIENT_ACCUMULATION_STEPS = 1  # No accumulation needed with large batch size

# Inference
BEAM_SIZE = 3
MAX_DECODING_STEPS = 10000

# LilyPond
LILYPOND_CMD = "lilypond"

# Checkpointing
CHECKPOINT_DIR = os.getenv("P2L_CKPT_DIR", "./checkpoints")

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"