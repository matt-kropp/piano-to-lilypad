import os
import torch

# Data paths
DATA_DIR = os.getenv("P2L_DATA_DIR", "/content/drive/MyDrive/p2l/data")
MAESTRO_DIR = os.path.join(DATA_DIR, "maestro-v3.0.0")  # Expect audio + MIDI pairs
SYNTHTRAIN_DIR = os.path.join(DATA_DIR, "synthetic")  # Synthetic MIDI renders

# Audio processing
SAMPLE_RATE = 44100
N_MELS = 229
HOP_LENGTH = 512  # ~11.6ms
WIN_LENGTH = 2048  # ~46ms

# Model architecture (A100 optimized - conservative memory)
# Option 1: Full A100 setup (~180M params)
HIDDEN_DIM = 1024
NUM_ENCODER_LAYERS = 16
NUM_DECODER_LAYERS = 16
NUM_ATTENTION_HEADS = 16
FEEDFORWARD_DIM = 4096
DROPOUT = 0.1

# Option 2: Medium setup if memory issues (~90M params)
# HIDDEN_DIM = 768
# NUM_ENCODER_LAYERS = 12
# NUM_DECODER_LAYERS = 12
# NUM_ATTENTION_HEADS = 12
# FEEDFORWARD_DIM = 3072

# Option 3: Conservative setup (~45M params)
# HIDDEN_DIM = 512
# NUM_ENCODER_LAYERS = 8
# NUM_DECODER_LAYERS = 8
# NUM_ATTENTION_HEADS = 8
# FEEDFORWARD_DIM = 2048

# Training (Optimized for A100 with memory constraints)
BATCH_SIZE = 16  # Keeping reasonable size for A100 efficiency
LEARNING_RATE = 2e-4  # Slightly reduced due to smaller effective batch
NUM_EPOCHS = 100
WARMUP_STEPS = 2000
WEIGHT_DECAY = 1e-2
TOTAL_STEPS = 100000  # Reduced since more efficient training
AUX_CTC_WEIGHT = 0.3

# Memory management (A100 optimized - conservative)
MAX_AUDIO_LENGTH = 75000  # Reduced from 150k (~40 seconds of audio)
GRADIENT_ACCUMULATION_STEPS = 2  # Accumulate 2 steps to maintain effective batch size of 32

# Inference
BEAM_SIZE = 3
MAX_DECODING_STEPS = 10000

# LilyPond
LILYPOND_CMD = "lilypond"

# Checkpointing
CHECKPOINT_DIR = os.getenv("P2L_CKPT_DIR", "/content/drive/MyDrive/p2l/checkpoints")

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"