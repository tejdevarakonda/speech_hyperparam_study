from pathlib import Path
import numpy as np
import soundfile as sf
import torch


# -------------------------------
# Load audio
# -------------------------------
def load_wav(path):
    audio, sr = sf.read(path, always_2d=False)

    # convert stereo → mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio.astype(np.float32), sr


# -------------------------------
# Save audio
# -------------------------------
def save_wav(path, audio, sr=16000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(path), audio, sr)


# -------------------------------
# Pad or crop audio
# -------------------------------
def pad_or_crop(audio, target_len, random_crop=False):
    audio = np.asarray(audio, dtype=np.float32)
    length = len(audio)

    # exact size
    if length == target_len:
        return audio

    # longer → crop
    if length > target_len:
        if random_crop:
            start = np.random.randint(0, length - target_len)
        else:
            start = (length - target_len) // 2

        return audio[start:start + target_len]

    # shorter → pad
    pad_len = target_len - length
    return np.pad(audio, (0, pad_len), mode="constant")


# -------------------------------
# Convert to tensor
# -------------------------------
def to_tensor(audio):
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)


# -------------------------------
# Ensure directory exists
# -------------------------------
def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Normalize audio (optional reuse)
# -------------------------------
def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio.astype(np.float32)