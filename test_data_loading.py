#!/usr/bin/env python3
"""
Test script to verify data loading works properly.
Run this before training to catch data loading issues early.
"""

import torch
from torch.utils.data import DataLoader
from src.piano_to_lilypond.dataset import PianoDataset
from src.piano_to_lilypond.train import get_file_list
from src.piano_to_lilypond.config import MAESTRO_DIR, BATCH_SIZE

def test_data_loading():
    print("🧪 Testing data loading...")
    
    # Get a small subset of files for testing
    all_pairs = get_file_list(MAESTRO_DIR)
    test_pairs = all_pairs[:10]  # Just test with 10 files
    
    print(f"Testing with {len(test_pairs)} files")
    
    # Test with no workers first
    print("Testing single-threaded loading...")
    dataset = PianoDataset(test_pairs)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
    try:
        for i, (src, tgt) in enumerate(loader):
            print(f"Batch {i+1}: src shape {src.shape}, tgt shape {tgt.shape}")
            if i >= 2:  # Just test a few batches
                break
        print("✅ Single-threaded loading successful!")
    except Exception as e:
        print(f"❌ Single-threaded loading failed: {e}")
        return False
    
    # Test with workers if single-threaded works
    print("Testing multi-threaded loading...")
    try:
        loader_mt = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
        for i, (src, tgt) in enumerate(loader_mt):
            print(f"MT Batch {i+1}: src shape {src.shape}, tgt shape {tgt.shape}")
            if i >= 2:
                break
        print("✅ Multi-threaded loading successful!")
        return True
    except Exception as e:
        print(f"⚠️ Multi-threaded loading failed: {e}")
        print("Will use single-threaded loading for training.")
        return True  # Single-threaded works, so we can proceed

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("🎉 Data loading test passed! Ready for training.")
    else:
        print("❌ Data loading test failed. Please check your data.") 