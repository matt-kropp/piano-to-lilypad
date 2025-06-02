#!/usr/bin/env python3
"""
Test script to verify windowed data loading works properly.
Run this before training to catch data loading issues early.
"""

import sys
import os
import time
sys.path.insert(0, 'src')

from torch.utils.data import DataLoader
from piano_to_lilypond.dataset import PianoDataset
from piano_to_lilypond.config import MAESTRO_DIR, BATCH_SIZE
import glob

def test_data_loading():
    print("🧪 Testing windowed data loading...")
    
    # Get fewer test files since windowing multiplies examples
    audio_files = glob.glob(os.path.join(MAESTRO_DIR, "**/*.wav"), recursive=True)[:5]
    data_list = []
    
    for audio_file in audio_files:
        # Find corresponding MIDI file
        midi_file = audio_file.replace('.wav', '.midi')
        if not os.path.exists(midi_file):
            midi_file = audio_file.replace('.wav', '.mid')
        
        if os.path.exists(midi_file):
            data_list.append((audio_file, midi_file))
    
    if not data_list:
        print("❌ No matching audio/MIDI pairs found!")
        return False
    
    print(f"Testing with {len(data_list)} files")
    print("Creating windowed dataset...")
    
    # Time the dataset creation to show caching benefit
    start_time = time.time()
    dataset = PianoDataset(data_list)
    creation_time = time.time() - start_time
    
    print(f"Dataset created: {len(dataset)} windows from {len(data_list)} files")
    print(f"Windows per file average: {len(dataset) / len(data_list):.1f}")
    print(f"Dataset creation time: {creation_time:.1f} seconds")
    
    # Show cache info
    dataset.cache_info()
    
    # Test cache loading by creating dataset again
    print("\n🔄 Testing cache loading...")
    start_time = time.time()
    dataset2 = PianoDataset(data_list)  # Should load from cache
    cache_time = time.time() - start_time
    print(f"Cache load time: {cache_time:.1f} seconds")
    
    if cache_time < creation_time / 2:
        print("✅ Cache significantly faster than creation!")
    else:
        print("⚠️ Cache not much faster, might not be working")
    
    # Test single-threaded loading
    print("\nTesting single-threaded loading...")
    try:
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, 
                          collate_fn=PianoDataset.collate_fn)
        
        for i, (src, tgt) in enumerate(loader):
            print(f"✅ Batch {i}: src shape {src.shape}, tgt shape {tgt.shape}")
            print(f"   Audio frames: {src.shape[1]}, MIDI tokens: {tgt.shape[1]}")
            if i >= 2:  # Test a few batches
                break
        print("✅ Single-threaded loading successful!")
        
    except Exception as e:
        print(f"❌ Single-threaded loading failed: {e}")
        print("Trying individual samples...")
        
        # Debug individual samples (windows)
        for i in range(min(5, len(dataset))):
            try:
                src, tgt = dataset[i]
                print(f"Window {i}: src shape {src.shape}, tgt shape {tgt.shape}")
            except Exception as sample_e:
                print(f"Window {i} failed: {sample_e}")
        
        return False
    
    # Test multi-threaded loading
    print("Testing multi-threaded loading...")
    try:
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, 
                          collate_fn=PianoDataset.collate_fn)
        
        for i, (src, tgt) in enumerate(loader):
            print(f"✅ Batch {i}: src shape {src.shape}, tgt shape {tgt.shape}")
            if i >= 2:
                break
        print("✅ Multi-threaded loading successful!")
        return True
        
    except Exception as e:
        print(f"⚠️ Multi-threaded loading failed: {e}")
        print("But single-threaded works, so training can proceed with num_workers=0")
        return True

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("🎉 Windowed data loading test passed!")
        print("Ready for training with consistent memory usage!")
        print("\n💡 Tip: Subsequent runs will be much faster due to caching!")
    else:
        print("❌ Data loading test failed. Please check your data.")
        sys.exit(1) 