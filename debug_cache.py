#!/usr/bin/env python3
"""
Debug why cache isn't being used in training.
"""

import sys
import os
sys.path.insert(0, 'src')

from piano_to_lilypond.dataset import PianoDataset
from piano_to_lilypond.train import get_file_list
from piano_to_lilypond.config import MAESTRO_DIR, MAX_MIDI_LENGTH
import random

def debug_cache():
    print("🔍 Debugging cache usage...")
    print("=" * 50)
    
    # Replicate exactly what training does
    maestro_pairs = get_file_list(MAESTRO_DIR)
    print(f"Found {len(maestro_pairs)} audio-MIDI pairs")
    
    random.shuffle(maestro_pairs)
    split = int(0.95 * len(maestro_pairs))
    train_pairs = maestro_pairs[:split]
    val_pairs = maestro_pairs[split:]
    
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print()
    
    # Check current working directory
    print(f"Current working directory: {os.getcwd()}")
    print(f"Cache directory should be: {os.path.join(os.getcwd(), '.dataset_cache')}")
    print(f"Cache directory exists: {os.path.exists('.dataset_cache')}")
    if os.path.exists('.dataset_cache'):
        cache_files = [f for f in os.listdir('.dataset_cache') if f.endswith('.pkl')]
        print(f"Cache files found: {cache_files}")
    print()
    
    # Create dataset exactly like training does
    print("🚀 Creating training dataset...")
    train_dataset = PianoDataset(train_pairs, max_seq_len=MAX_MIDI_LENGTH)
    print(f"Training dataset length: {len(train_dataset)}")
    print()
    
    print("🚀 Creating validation dataset...")
    val_dataset = PianoDataset(val_pairs, max_seq_len=MAX_MIDI_LENGTH)
    print(f"Validation dataset length: {len(val_dataset)}")
    print()
    
    # Show cache info
    print("📊 Cache information:")
    train_dataset.cache_info()
    val_dataset.cache_info()

if __name__ == "__main__":
    debug_cache() 