#!/usr/bin/env python3
"""
Test script to verify data loading works properly.
Run this before training to catch data loading issues early.
"""

import sys
import os
sys.path.insert(0, 'src')

from torch.utils.data import DataLoader
from piano_to_lilypond.dataset import PianoDataset
from piano_to_lilypond.config import MAESTRO_DIR
import glob

def test_data_loading():
    print("🧪 Testing data loading...")
    
    # Get some test files
    audio_files = glob.glob(os.path.join(MAESTRO_DIR, "**/*.wav"), recursive=True)[:10]
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
    dataset = PianoDataset(data_list, max_seq_len=1000)
    
    # Test single-threaded loading
    print("Testing single-threaded loading...")
    try:
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, 
                          collate_fn=PianoDataset.collate_fn)
        
        for i, (src, tgt) in enumerate(loader):
            print(f"✅ Batch {i}: src shape {src.shape}, tgt shape {tgt.shape}")
            if i >= 2:  # Test a few batches
                break
        print("✅ Single-threaded loading successful!")
        
    except Exception as e:
        print(f"❌ Single-threaded loading failed: {e}")
        print("Trying individual samples...")
        
        # Debug individual samples
        for i in range(min(3, len(dataset))):
            try:
                src, tgt = dataset[i]
                print(f"Sample {i}: src shape {src.shape}, tgt shape {tgt.shape}")
            except Exception as sample_e:
                print(f"Sample {i} failed: {sample_e}")
        
        return False
    
    # Test multi-threaded loading
    print("Testing multi-threaded loading...")
    try:
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2, 
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
        print("🎉 Data loading test passed!")
    else:
        print("❌ Data loading test failed. Please check your data.")
        sys.exit(1) 