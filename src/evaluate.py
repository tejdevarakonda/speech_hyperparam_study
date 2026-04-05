import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pystoi.stoi import stoi

from dataset import VoiceBankDataset
from model import SpeechEnhancementUNet
from utils import save_wav


def si_sdr(reference, estimation, eps=1e-8):
    reference = reference.astype(np.float32)
    estimation = estimation.astype(np.float32)

    reference = reference - np.mean(reference)
    estimation = estimation - np.mean(estimation)

    ref_energy = np.sum(reference ** 2) + eps
    scale = np.sum(estimation * reference) / ref_energy
    target = scale * reference
    noise = estimation - target

    ratio = (np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps)
    return 10.0 * np.log10(ratio + eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="data/metadata/test_processed_pairs.csv")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--segment_seconds", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    enhanced_dir = out_dir / "enhanced_audio"
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    ds = VoiceBankDataset(
        args.test_csv,
        segment_seconds=args.segment_seconds,
        train=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SpeechEnhancementUNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []

    with torch.no_grad():
        for idx, (noisy, clean) in enumerate(loader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            enhanced = model(noisy)

            noisy_np = noisy.squeeze(0).squeeze(0).cpu().numpy()
            clean_np = clean.squeeze(0).squeeze(0).cpu().numpy()
            enh_np = enhanced.squeeze(0).squeeze(0).cpu().numpy()

            stoi_score = stoi(clean_np, enh_np, 16000, extended=False)
            sisdr_score = si_sdr(clean_np, enh_np)

            rows.append(
                {
                    "index": idx,
                    "stoi": stoi_score,
                    "si_sdr": sisdr_score,
                }
            )

            save_wav(enhanced_dir / f"{idx:04d}.wav", enh_np, sr=16000)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_metrics.csv", index=False)

    print("Evaluation complete.")
    print("Mean STOI:", df["stoi"].mean())
    print("Mean SI-SDR:", df["si_sdr"].mean())
    print("Saved:", out_dir / "test_metrics.csv")


if __name__ == "__main__":
    main()