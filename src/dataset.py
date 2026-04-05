import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import load_wav, to_tensor


class VoiceBankDataset(Dataset):
    def __init__(self, csv_path, segment_seconds=2.0, sample_rate=16000, train=True):
        self.df = pd.read_csv(csv_path)
        self.segment_len = int(segment_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.train = train

    def __len__(self):
        return len(self.df)

    def pad_or_crop_pair(self, noisy, clean):
        noisy_len = len(noisy)
        clean_len = len(clean)

        # make sure both are same length first
        min_len = min(noisy_len, clean_len)
        noisy = noisy[:min_len]
        clean = clean[:min_len]

        length = len(noisy)

        if length >= self.segment_len:
            if self.train:
                start = np.random.randint(0, length - self.segment_len + 1)
            else:
                start = (length - self.segment_len) // 2

            end = start + self.segment_len
            return noisy[start:end], clean[start:end]

        # if shorter than segment_len, pad both
        pad_len = self.segment_len - length
        noisy = np.pad(noisy, (0, pad_len), mode="constant")
        clean = np.pad(clean, (0, pad_len), mode="constant")
        return noisy, clean

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        noisy, sr1 = load_wav(row["noisy_processed"])
        clean, sr2 = load_wav(row["clean_processed"])

        if sr1 != self.sample_rate:
            raise ValueError(f"Unexpected sample rate in noisy file: {sr1}")
        if sr2 != self.sample_rate:
            raise ValueError(f"Unexpected sample rate in clean file: {sr2}")

        noisy, clean = self.pad_or_crop_pair(noisy, clean)

        noisy = to_tensor(noisy)
        clean = to_tensor(clean)

        return noisy, clean