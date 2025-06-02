#!/usr/bin/env python3
"""
Inspect the dataset cache to verify it contains proper window metadata.
"""

import pickle
import os
import glob

def inspect_cache():
    cache_files = glob.glob('.dataset_cache/*.pkl')
    
    if not cache_files:
        print("❌ No cache files found in .dataset_cache/")
        return
    
    for cache_file in cache_files:
        print(f"🔍 Inspecting: {cache_file}")
        size_kb = os.path.getsize(cache_file) / 1024
        print(f"📏 File size: {size_kb:.1f} KB")
        
        try:
            with open(cache_file, 'rb') as f:
                windows = pickle.load(f)
            
            print(f"📊 Total windows: {len(windows):,}")
            
            if windows:
                # Show first few windows
                print(f"\n📝 Sample windows:")
                for i, window in enumerate(windows[:3]):
                    audio_path, midi_path, start, end = window
                    duration_sec = (end - start) / 44100  # Assuming 44.1kHz
                    print(f"  Window {i}: {os.path.basename(audio_path)}")
                    print(f"    Duration: {duration_sec:.1f}s ({start:,} to {end:,} samples)")
                
                # Calculate estimated total duration
                if len(windows) > 0:
                    sample_duration = (windows[0][3] - windows[0][2]) / 44100
                    total_hours = len(windows) * sample_duration / 3600
                    print(f"\n⏱️ Estimated total training data: {total_hours:.1f} hours")
                
        except Exception as e:
            print(f"❌ Error reading cache: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    inspect_cache() 