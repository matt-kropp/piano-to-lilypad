#!/usr/bin/env python3
"""
Build dataset cache on CPU instance using parallel processing.
Much faster than sequential processing for large datasets.
"""

import sys
import os
import time
import multiprocessing as mp
from functools import partial
sys.path.insert(0, 'src')

from piano_to_lilypond.dataset import PianoDataset
from piano_to_lilypond.config import MAESTRO_DIR, WINDOW_SIZE_SECONDS, WINDOW_OVERLAP_SECONDS, SAMPLE_RATE
from piano_to_lilypond.utils.audio_utils import load_audio

def process_single_file(file_pair, window_samples, step_samples):
    """Process a single audio file to extract window information"""
    audio_path, midi_path = file_pair
    windows = []
    
    try:
        # Quick check - skip obviously huge files
        audio_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        if audio_size > 1000:  # Skip files larger than 1GB
            return windows
            
        # Load audio to determine length
        wav = load_audio(audio_path, SAMPLE_RATE)
        audio_length = len(wav)
        
        # Create windows for this file
        start = 0
        while start + window_samples <= audio_length:
            end = start + window_samples
            windows.append((audio_path, midi_path, start, end))
            start += step_samples
            
        # Add final partial window if significant length remains
        if start < audio_length and (audio_length - start) > window_samples // 2:
            windows.append((audio_path, midi_path, audio_length - window_samples, audio_length))
            
    except Exception as e:
        print(f"Warning: Failed to process {audio_path}: {e}")
    
    return windows

def build_cache_parallel():
    print("🏗️ Building dataset cache with parallel processing...")
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
    random.seed(42)  # Fixed seed for consistent splits
    random.shuffle(maestro_pairs)
    split = int(0.95 * len(maestro_pairs))
    train_pairs = maestro_pairs[:split]
    val_pairs = maestro_pairs[split:]
    
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print("=" * 50)
    
    # Calculate window parameters
    window_samples = int(WINDOW_SIZE_SECONDS * SAMPLE_RATE)
    overlap_samples = int(WINDOW_OVERLAP_SECONDS * SAMPLE_RATE)
    step_samples = window_samples - overlap_samples
    
    # Determine number of processes (leave some cores free)
    num_processes = max(1, mp.cpu_count() - 1)
    print(f"Using {num_processes} parallel processes")
    print("=" * 50)
    
    # Build training cache in parallel
    print("🚀 Building TRAINING dataset cache in parallel...")
    start_time = time.time()
    
    # Create partial function with fixed parameters
    process_func = partial(process_single_file, 
                          window_samples=window_samples, 
                          step_samples=step_samples)
    
    # Process files in parallel
    with mp.Pool(processes=num_processes) as pool:
        # Show progress
        results = []
        chunk_size = max(1, len(train_pairs) // (num_processes * 4))  # Good chunk size
        
        print(f"Processing {len(train_pairs)} training files in chunks of {chunk_size}...")
        
        # Submit all work
        for i in range(0, len(train_pairs), chunk_size):
            chunk = train_pairs[i:i + chunk_size]
            result = pool.map_async(process_func, chunk)
            results.append(result)
        
        # Collect results with progress
        all_windows = []
        total_chunks = len(results)
        
        for i, result in enumerate(results):
            chunk_windows = result.get()  # This blocks until chunk is done
            # Flatten the list of lists
            for file_windows in chunk_windows:
                all_windows.extend(file_windows)
            
            progress = (i + 1) / total_chunks * 100
            print(f"Progress: {progress:.1f}% ({i+1}/{total_chunks} chunks completed)")
    
    train_time = time.time() - start_time
    
    # Save training cache manually (since we bypassed PianoDataset)
    import pickle
    import hashlib
    
    # Create cache directory
    cache_dir = os.path.join(os.getcwd(), '.dataset_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache path (similar to PianoDataset logic)
    config_str = f"{len(train_pairs)}_{WINDOW_SIZE_SECONDS}_{WINDOW_OVERLAP_SECONDS}"
    if train_pairs:
        config_str += f"_{train_pairs[0][0]}_{train_pairs[-1][0]}"
    cache_hash = hashlib.md5(config_str.encode()).hexdigest()[:16]
    train_cache_path = os.path.join(cache_dir, f"windows_cache_{cache_hash}.pkl")
    
    with open(train_cache_path, 'wb') as f:
        pickle.dump(all_windows, f)
    
    print(f"✅ Training cache built: {len(all_windows):,} windows")
    print(f"⏱️ Training cache time: {train_time:.1f} seconds")
    print(f"💾 Saved to: {train_cache_path}")
    print()
    
    # Build validation cache in parallel
    print("🚀 Building VALIDATION dataset cache in parallel...")
    start_time = time.time()
    
    with mp.Pool(processes=num_processes) as pool:
        chunk_size = max(1, len(val_pairs) // (num_processes * 2))
        print(f"Processing {len(val_pairs)} validation files in chunks of {chunk_size}...")
        
        # Process validation files
        val_results = []
        for i in range(0, len(val_pairs), chunk_size):
            chunk = val_pairs[i:i + chunk_size]
            result = pool.map_async(process_func, chunk)
            val_results.append(result)
        
        # Collect validation results
        val_windows = []
        for i, result in enumerate(val_results):
            chunk_windows = result.get()
            for file_windows in chunk_windows:
                val_windows.extend(file_windows)
            
            progress = (i + 1) / len(val_results) * 100
            print(f"Validation progress: {progress:.1f}% ({i+1}/{len(val_results)} chunks)")
    
    val_time = time.time() - start_time
    
    # Save validation cache
    val_config_str = f"{len(val_pairs)}_{WINDOW_SIZE_SECONDS}_{WINDOW_OVERLAP_SECONDS}"
    if val_pairs:
        val_config_str += f"_{val_pairs[0][0]}_{val_pairs[-1][0]}"
    val_cache_hash = hashlib.md5(val_config_str.encode()).hexdigest()[:16]
    val_cache_path = os.path.join(cache_dir, f"windows_cache_{val_cache_hash}.pkl")
    
    with open(val_cache_path, 'wb') as f:
        pickle.dump(val_windows, f)
    
    print(f"✅ Validation cache built: {len(val_windows):,} windows")
    print(f"⏱️ Validation cache time: {val_time:.1f} seconds")
    print(f"💾 Saved to: {val_cache_path}")
    print()
    
    # Summary
    total_windows = len(all_windows) + len(val_windows)
    total_time = train_time + val_time
    
    print("=" * 50)
    print("📊 PARALLEL CACHE BUILD SUMMARY")
    print("=" * 50)
    print(f"Processes used: {num_processes}")
    print(f"Total files processed: {len(maestro_pairs)}")
    print(f"Total windows created: {total_windows:,}")
    print(f"Windows per file: {total_windows/len(maestro_pairs):.1f}")
    print(f"Total build time: {total_time:.1f} seconds")
    print(f"Speed: {len(maestro_pairs)/total_time:.1f} files/second")
    print(f"Cache directory: {cache_dir}")
    
    # Show cache files
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
    # Protect against multiprocessing issues on some systems
    mp.set_start_method('spawn', force=True)
    
    success = build_cache_parallel()
    if success:
        print("\n🎉 Parallel cache building completed successfully!")
        print("Ready to transfer to A100 instance!")
    else:
        print("\n❌ Cache building failed.")
        sys.exit(1) 