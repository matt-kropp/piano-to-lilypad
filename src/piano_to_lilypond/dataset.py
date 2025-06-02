import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import hashlib
import json
import multiprocessing as mp
from functools import partial
from .utils.audio_utils import load_audio, compute_mel_spectrogram
from .utils.midi_utils import midi_to_token_sequence, token_to_id, build_vocab
from .config import (SAMPLE_RATE, N_MELS, HOP_LENGTH, WIN_LENGTH, MAX_AUDIO_LENGTH, 
                     WINDOW_SIZE_SECONDS, WINDOW_OVERLAP_SECONDS, MAX_MIDI_LENGTH)

def _process_single_file_for_cache(file_pair, window_samples, step_samples):
    """Process a single audio file to extract window information (for multiprocessing)"""
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

class PianoDataset(Dataset):
    def __init__(self, data_list, max_seq_len=None, max_audio_len=None, cache_dir=None):
        """
        data_list: list of tuples (audio_path, midi_path)
        Creates windowed examples from each audio/MIDI pair
        """
        self.data_list = data_list
        self.max_seq_len = max_seq_len or MAX_MIDI_LENGTH
        self.max_audio_len = max_audio_len or MAX_AUDIO_LENGTH
        self.window_samples = int(WINDOW_SIZE_SECONDS * SAMPLE_RATE)
        self.overlap_samples = int(WINDOW_OVERLAP_SECONDS * SAMPLE_RATE)
        self.step_samples = self.window_samples - self.overlap_samples
        
        # Setup caching
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), '.dataset_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Pre-compute all windows for efficient indexing (with caching)
        self.windows = self._create_windows_cached()
        print(f"Created {len(self.windows)} windowed examples from {len(data_list)} files")

    def _get_cache_path(self):
        """Generate cache file path based on dataset configuration"""
        # Create a hash of the dataset configuration
        config_str = f"{len(self.data_list)}_{WINDOW_SIZE_SECONDS}_{WINDOW_OVERLAP_SECONDS}"
        # Include first and last file paths to detect dataset changes
        if self.data_list:
            config_str += f"_{self.data_list[0][0]}_{self.data_list[-1][0]}"
        cache_hash = hashlib.md5(config_str.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"windows_cache_{cache_hash}.pkl")

    def _is_cache_valid(self, cache_path):
        """Check if cache is still valid (files haven't changed)"""
        if not os.path.exists(cache_path):
            return False
        
        try:
            cache_mtime = os.path.getmtime(cache_path)
            
            # Check if any source files are newer than cache
            for audio_path, midi_path in self.data_list[:10]:  # Sample check for speed
                if os.path.exists(audio_path) and os.path.getmtime(audio_path) > cache_mtime:
                    return False
                if os.path.exists(midi_path) and os.path.getmtime(midi_path) > cache_mtime:
                    return False
            return True
        except:
            return False

    def _create_windows_cached(self):
        """Create windows with caching support"""
        cache_path = self._get_cache_path()
        
        # Try to load from cache
        if self._is_cache_valid(cache_path):
            try:
                print(f"🚀 Loading windowed dataset from cache: {os.path.basename(cache_path)}")
                with open(cache_path, 'rb') as f:
                    windows = pickle.load(f)
                print(f"✅ Loaded {len(windows):,} windows from cache")
                return windows
            except Exception as e:
                print(f"⚠️ Failed to load cache: {e}, regenerating...")

        # Generate windows and cache them
        print(f"🏗️ Cache not found - creating windowed dataset with parallel processing...")
        print(f"📁 Will cache to: {os.path.basename(cache_path)}")
        
        import time
        start_time = time.time()
        windows = self._create_windows()
        build_time = time.time() - start_time
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(windows, f)
            cache_size = os.path.getsize(cache_path) / (1024 * 1024)  # MB
            print(f"💾 Cached windowed dataset: {os.path.basename(cache_path)} ({cache_size:.1f} MB)")
            print(f"⚡ Cache build time: {build_time:.1f} seconds")
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
        
        return windows

    def _create_windows(self):
        """Pre-compute all valid windows from the dataset using parallel processing"""
        print(f"Building cache in parallel for {len(self.data_list)} files...")
        
        # Calculate window parameters
        window_samples = int(WINDOW_SIZE_SECONDS * SAMPLE_RATE)
        overlap_samples = int(WINDOW_OVERLAP_SECONDS * SAMPLE_RATE)
        step_samples = window_samples - overlap_samples
        
        # Determine number of processes (leave some cores free)
        num_processes = max(1, mp.cpu_count() - 1)
        print(f"Using {num_processes} parallel processes for cache building")
        
        # Create partial function with fixed parameters
        process_func = partial(_process_single_file_for_cache, 
                              window_samples=window_samples, 
                              step_samples=step_samples)
        
        all_windows = []
        
        try:
            # Process files in parallel
            with mp.Pool(processes=num_processes) as pool:
                # Process in chunks for better progress tracking
                chunk_size = max(1, len(self.data_list) // (num_processes * 4))
                
                print(f"Processing files in chunks of {chunk_size}...")
                
                # Submit all work
                results = []
                for i in range(0, len(self.data_list), chunk_size):
                    chunk = self.data_list[i:i + chunk_size]
                    result = pool.map_async(process_func, chunk)
                    results.append(result)
                
                # Collect results with progress
                total_chunks = len(results)
                
                for i, result in enumerate(results):
                    chunk_windows = result.get()  # This blocks until chunk is done
                    # Flatten the list of lists
                    for file_windows in chunk_windows:
                        all_windows.extend(file_windows)
                    
                    progress = (i + 1) / total_chunks * 100
                    print(f"Cache building progress: {progress:.1f}% ({i+1}/{total_chunks} chunks completed)")
                    
        except Exception as e:
            print(f"Parallel processing failed: {e}")
            print("Falling back to sequential processing...")
            
            # Fallback to sequential processing
            for i, (audio_path, midi_path) in enumerate(self.data_list):
                if i % 100 == 0:  # Progress indicator
                    print(f"Processing file {i+1}/{len(self.data_list)}: {os.path.basename(audio_path)}")
                    
                try:
                    # Quick check - skip obviously huge files
                    audio_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
                    if audio_size > 1000:  # Skip files larger than 1GB
                        continue
                        
                    # Load audio to determine length
                    wav = load_audio(audio_path, SAMPLE_RATE)
                    audio_length = len(wav)
                    
                    # Create windows for this file
                    start = 0
                    while start + window_samples <= audio_length:
                        end = start + window_samples
                        all_windows.append((audio_path, midi_path, start, end))
                        start += step_samples
                        
                    # Add final partial window if significant length remains
                    if start < audio_length and (audio_length - start) > window_samples // 2:
                        all_windows.append((audio_path, midi_path, audio_length - window_samples, audio_length))
                        
                except Exception as e:
                    print(f"Warning: Failed to process {audio_path}: {e}")
                    continue
        
        print(f"✅ Cache building completed: {len(all_windows):,} windows created")
        return all_windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        audio_path, midi_path, start_sample, end_sample = self.windows[idx]
        
        try:
            # Load audio window
            wav = load_audio(audio_path, SAMPLE_RATE)
            wav_window = wav[start_sample:end_sample]
            
            # Pad if needed (for final partial windows)
            if len(wav_window) < self.window_samples:
                padding = self.window_samples - len(wav_window)
                wav_window = np.pad(wav_window, (0, padding), mode='constant')
            
            # Compute mel spectrogram
            mel = compute_mel_spectrogram(wav_window, SAMPLE_RATE, N_MELS, HOP_LENGTH, WIN_LENGTH)
            T = mel.shape[1]
            
            # Stack ±2 frames for context window
            stacked = []
            stack_size = 5
            for t in range(T):
                frames = []
                for offset in range(-(stack_size//2), (stack_size//2)+1):
                    tt = min(max(t + offset, 0), T - 1)
                    frames.append(mel[:, tt])
                stacked.append(np.stack(frames, axis=0))  # [5, N_MELS]
            src = np.stack(stacked, axis=0)  # [T, 5, N_MELS]
            
            # Convert to torch.FloatTensor
            src = torch.from_numpy(src).float()
            del mel, stacked  # Clean up

            # Load MIDI and extract corresponding time window
            token_seq = midi_to_token_sequence(midi_path)
            
            # For windowing, we take the full MIDI sequence but limit its length
            # In practice, you might want to align MIDI timing with audio windows
            # For now, we'll just use the full sequence truncated to max length
            tgt = [token_to_id[t] for t in token_seq if t in token_to_id]
            if len(tgt) > self.max_seq_len:
                # Take a random chunk of the MIDI sequence
                if len(tgt) > self.max_seq_len:
                    start_idx = random.randint(0, len(tgt) - self.max_seq_len)
                    tgt = tgt[start_idx:start_idx + self.max_seq_len]
            
            # Ensure minimum sequence length
            if len(tgt) < 10:
                tgt = [token_to_id['EOS']] * 10
                
            tgt = torch.LongTensor(tgt)

            return src, tgt
            
        except Exception as e:
            # Fallback to dummy data
            print(f"Warning: Error processing window {idx}: {e}")
            dummy_src = torch.zeros(self.max_audio_len // 4, 5, N_MELS)  # Rough estimate of mel frames
            dummy_tgt = torch.LongTensor([token_to_id['EOS']] * 10)
            return dummy_src, dummy_tgt

    @staticmethod
    def collate_fn(batch):
        # batch: list of (src, tgt)
        srcs, tgts = zip(*batch)
        
        # Pad tgt sequences
        tgt_lens = [len(t) for t in tgts]
        max_tgt_len = max(tgt_lens)
        padded_tgts = torch.full((len(tgts), max_tgt_len), token_to_id['PAD'], dtype=torch.long)
        for i, t in enumerate(tgts):
            padded_tgts[i, :len(t)] = t
        
        # Pad src sequences - varying length in time dimension
        src_lens = [s.shape[0] for s in srcs]
        max_src_len = max(src_lens)
        
        # Get dimensions from first tensor: (T, 5, N_MELS)
        B = len(srcs)
        stack_size = srcs[0].shape[1]  # Should be 5
        n_mels = srcs[0].shape[2]      # Should be N_MELS (229)
        
        # Create padded tensor: (B, T, 5, N_MELS)
        stacked = torch.zeros((B, max_src_len, stack_size, n_mels), dtype=torch.float)
        for i, s in enumerate(srcs):
            T = s.shape[0]
            stacked[i, :T] = s
            
        return stacked, padded_tgts

    def clear_cache(self):
        """Clear the dataset cache"""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"🗑️ Cleared cache: {cache_path}")
        else:
            print("No cache to clear")

    def cache_info(self):
        """Display cache information"""
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            mtime = os.path.getmtime(cache_path)
            print(f"📁 Cache file: {cache_path}")
            print(f"📏 Cache size: {size_mb:.1f} MB")
            print(f"🕒 Cache created: {os.path.ctime(mtime)}")
        else:
            print("No cache file exists")

    @staticmethod
    def clear_all_caches(cache_dir=None):
        """Clear all dataset caches"""
        cache_dir = cache_dir or os.path.join(os.getcwd(), '.dataset_cache')
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print(f"🗑️ Cleared all caches in: {cache_dir}")
        else:
            print("No cache directory exists")