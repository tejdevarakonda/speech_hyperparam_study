from pathlib import Path
import math
import pandas as pd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000

META_DIR = Path("data/metadata")
PROCESSED_DIR = Path("data/processed")

INPUT_CSVS = {
    "train": META_DIR / "train_pairs.csv",
    "val": META_DIR / "val_pairs.csv",
    "test": META_DIR / "test_pairs.csv",
}

OUTPUT_CSVS = {
    "train": META_DIR / "train_processed_pairs.csv",
    "val": META_DIR / "val_processed_pairs.csv",
    "test": META_DIR / "test_processed_pairs.csv",
}


def ensure_folders():
    for split in ["train", "val", "test"]:
        (PROCESSED_DIR / split / "clean").mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / split / "noisy").mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)


def load_audio(path):
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sr


def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio.astype(np.float32)


def process_one_file(src_path, dst_path):
    audio, sr = load_audio(src_path)
    audio = resample_audio(audio, sr, TARGET_SR)
    audio = normalize_audio(audio)
    sf.write(dst_path, audio, TARGET_SR)
    return str(dst_path.as_posix())


def process_split(split):
    in_csv = INPUT_CSVS[split]
    out_csv = OUTPUT_CSVS[split]

    df = pd.read_csv(in_csv)

    clean_out = []
    noisy_out = []

    for _, row in df.iterrows():
        utt_id = row["utt_id"]

        clean_src = Path(row["clean_path"])
        noisy_src = Path(row["noisy_path"])

        clean_dst = PROCESSED_DIR / split / "clean" / f"{utt_id}.wav"
        noisy_dst = PROCESSED_DIR / split / "noisy" / f"{utt_id}.wav"

        clean_out.append(process_one_file(clean_src, clean_dst))
        noisy_out.append(process_one_file(noisy_src, noisy_dst))

    df["clean_processed"] = clean_out
    df["noisy_processed"] = noisy_out

    df.to_csv(out_csv, index=False)
    print(f"Saved {split}: {len(df)} files -> {out_csv}")


def main():
    ensure_folders()

    for split in ["train", "val", "test"]:
        print(f"Processing {split}...")
        process_split(split)

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()