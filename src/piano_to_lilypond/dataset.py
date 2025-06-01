import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np
from .utils.audio_utils import load_audio, compute_mel_spectrogram
from .utils.midi_utils import midi_to_token_sequence, token_to_id, build_vocab
from .config import SAMPLE_RATE, N_MELS, HOP_LENGTH, WIN_LENGTH, MAX_AUDIO_LENGTH

class PianoDataset(Dataset):
    def __init__(self, data_list, max_seq_len=10000, max_audio_len=None):
        """
        data_list: list of tuples (audio_path, midi_path)
        max_seq_len: maximum length for MIDI token sequences
        max_audio_len: maximum length for audio sequences (in mel frames)
        """
        self.data_list = data_list
        self.max_seq_len = max_seq_len
        self.max_audio_len = max_audio_len or MAX_AUDIO_LENGTH

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        audio_path, midi_path = self.data_list[idx]
        
        try:
            wav = load_audio(audio_path, SAMPLE_RATE)
            mel = compute_mel_spectrogram(wav, SAMPLE_RATE, N_MELS, HOP_LENGTH, WIN_LENGTH)
            
            # Limit audio length to prevent memory issues
            T = mel.shape[1]
            if T > self.max_audio_len:
                # Take a random chunk from the audio
                start_idx = random.randint(0, T - self.max_audio_len)
                mel = mel[:, start_idx:start_idx + self.max_audio_len]
                T = self.max_audio_len
            
            # Skip very short files (less than 1 second)
            if T < 100:  # ~1 second at default settings
                raise ValueError(f"Audio too short: {T} frames")
                
            # Stack ±2 frames for context window if desired; pad at edges
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

            # MIDI → token IDs
            token_seq = midi_to_token_sequence(midi_path)
            tgt = [token_to_id[t] for t in token_seq if t in token_to_id]
            if len(tgt) > self.max_seq_len:
                tgt = tgt[:self.max_seq_len-1] + [token_to_id['EOS']]
            
            # Skip files with very short MIDI sequences
            if len(tgt) < 10:
                raise ValueError(f"MIDI sequence too short: {len(tgt)} tokens")
                
            tgt = torch.LongTensor(tgt)

            return src, tgt
            
        except Exception as e:
            # Return a fallback item or skip
            print(f"Warning: Skipping {audio_path}: {e}")
            # Return the first item as fallback (recursive call with idx=0)
            if idx != 0:
                return self.__getitem__(0)
            else:
                # Create minimal dummy data as absolute fallback
                dummy_src = torch.zeros(1000, 5, N_MELS)
                dummy_tgt = torch.LongTensor([token_to_id['EOS']])
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
        # srcs are varying length in time dimension; pad them as well
        src_lens = [s.shape[0] for s in srcs]
        max_src_len = max(src_lens)
        # src shape: (T, 5, N_MELS) → we want (B, T, 5, N_MELS)
        B = len(srcs)
        stacked = torch.zeros((B, max_src_len, 5, N_MELS), dtype=torch.float)
        for i, s in enumerate(srcs):
            T = s.shape[0]
            stacked[i, :T] = s
        return stacked, padded_tgts