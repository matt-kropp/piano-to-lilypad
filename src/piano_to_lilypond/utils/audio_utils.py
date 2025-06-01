import librosa
import numpy as np

def load_audio(path, sr):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def compute_mel_spectrogram(wav, sr, n_mels, hop_length, win_length):
    # Compute mel-spectrogram and return log-amplitude
    S = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_mels=n_mels, n_fft=win_length, hop_length=hop_length,
        fmin=27.5, fmax=sr // 2
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    # Normalize to zero mean, unit variance per mel-band
    mean = np.mean(S_db, axis=1, keepdims=True)
    std = np.std(S_db, axis=1, keepdims=True) + 1e-6
    S_norm = (S_db - mean) / std
    return S_norm.astype(np.float32)