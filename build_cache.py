#!/usr/bin/env python3
"""
Build dataset cache on CPU instance for later use on GPU.
This saves money by doing the preprocessing on cheaper CPU instances.
"""

import sys
import os
import time
sys.path.insert(0, 'src')

from piano_to_lilypond.dataset import PianoDataset
from piano_to_lilypond.config import MAESTRO_DIR

def build_cache():
    print("🏗️ Building dataset cache on CPU...")
    print("=" * 50)
    
    # Import train module to get file list function
    from piano_to_lilypond.train import get_file_list
    import random
    
    # Get file list
    maestro_pairs = get_file_list(MAESTRO_DIR)
    print(f"Found {len(maestro_pairs)} audio-MIDI pairs")
    
    if len(maestro_pairs) == 0:
        print("❌ No training data found! Please check MAESTRO_DIR path.")
        return False
    
    # Split into train/val (same as training script)
    random.shuffle(maestro_pairs)
    split = int(0.95 * len(maestro_pairs))
    train_pairs = maestro_pairs[:split]
    val_pairs = maestro_pairs[split:]
    
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print("=" * 50)
    
    # Build training cache
    print("🚀 Building TRAINING dataset cache...")
    start_time = time.time()
    train_dataset = PianoDataset(train_pairs)
    train_time = time.time() - start_time
    
    print(f"✅ Training cache built: {len(train_dataset)} windows")
    print(f"⏱️ Training cache time: {train_time:.1f} seconds")
    train_dataset.cache_info()
    print()
    
    # Build validation cache  
    print("🚀 Building VALIDATION dataset cache...")
    start_time = time.time()
    val_dataset = PianoDataset(val_pairs)
    val_time = time.time() - start_time
    
    print(f"✅ Validation cache built: {len(val_dataset)} windows")
    print(f"⏱️ Validation cache time: {val_time:.1f} seconds")
    val_dataset.cache_info()
    print()
    
    # Summary
    total_windows = len(train_dataset) + len(val_dataset)
    total_time = train_time + val_time
    
    print("=" * 50)
    print("📊 CACHE BUILD SUMMARY")
    print("=" * 50)
    print(f"Total files processed: {len(maestro_pairs)}")
    print(f"Total windows created: {total_windows:,}")
    print(f"Windows per file: {total_windows/len(maestro_pairs):.1f}")
    print(f"Total build time: {total_time:.1f} seconds")
    print(f"Cache directory: {train_dataset.cache_dir}")
    
    # Show cache files
    cache_dir = train_dataset.cache_dir
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
        total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cache_files)
        print(f"Cache files: {len(cache_files)}")
        print(f"Total cache size: {total_size / (1024*1024):.1f} MB")
        
        for f in cache_files:
            size_mb = os.path.getsize(os.path.join(cache_dir, f)) / (1024*1024)
            print(f"  - {f}: {size_mb:.1f} MB")
    
    print("\n💡 NEXT STEPS:")
    print("1. Copy the entire .dataset_cache/ folder to your A100 instance")
    print("2. Ensure the data paths are the same on A100")
    print("3. Training will start instantly using the cached windows!")
    
    return True

if __name__ == "__main__":
    success = build_cache()
    if success:
        print("\n🎉 Cache building completed successfully!")
        print("Ready to transfer to A100 instance!")
    else:
        print("\n❌ Cache building failed.")
        sys.exit(1) 