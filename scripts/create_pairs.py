from pathlib import Path
import pandas as pd
import random

RAW_DIR = Path("data/raw")
META_DIR = Path("data/metadata")
META_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CLEAN = RAW_DIR / "clean_trainset_28spk_wav"
TRAIN_NOISY = RAW_DIR / "noisy_trainset_28spk_wav"
TEST_CLEAN = RAW_DIR / "clean_testset_wav"
TEST_NOISY = RAW_DIR / "noisy_testset_wav"


def build_pairs(clean_dir: Path, noisy_dir: Path):
    """
    Match files ONLY by filename (ignoring folder structure)
    """
    clean_files = {}
    noisy_files = {}

    # collect all clean files
    for p in clean_dir.rglob("*.wav"):
        clean_files[p.name] = p

    # collect all noisy files
    for p in noisy_dir.rglob("*.wav"):
        noisy_files[p.name] = p

    common_names = sorted(set(clean_files.keys()) & set(noisy_files.keys()))

    rows = []
    for name in common_names:
        clean_path = clean_files[name]
        noisy_path = noisy_files[name]

        utt_id = Path(name).stem
        speaker_id = utt_id.split("_")[0] if "_" in utt_id else "unknown"

        rows.append({
            "utt_id": utt_id,
            "speaker_id": speaker_id,
            "clean_path": str(clean_path.as_posix()),
            "noisy_path": str(noisy_path.as_posix()),
        })

    return pd.DataFrame(rows)


def train_val_split(df, val_ratio=0.2, seed=42):
    rows = df.to_dict("records")
    random.Random(seed).shuffle(rows)

    n_val = int(len(rows) * val_ratio)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    return pd.DataFrame(train_rows), pd.DataFrame(val_rows)


def main():
    print("Checking files...")
    print("Train clean:", len(list(TRAIN_CLEAN.rglob("*.wav"))))
    print("Train noisy:", len(list(TRAIN_NOISY.rglob("*.wav"))))
    print("Test clean:", len(list(TEST_CLEAN.rglob("*.wav"))))
    print("Test noisy:", len(list(TEST_NOISY.rglob("*.wav"))))

    train_df = build_pairs(TRAIN_CLEAN, TRAIN_NOISY)
    test_df = build_pairs(TEST_CLEAN, TEST_NOISY)

    if len(train_df) == 0:
        print("\nStill no pairs found — something is wrong.")
        return

    train_df, val_df = train_val_split(train_df, val_ratio=0.2)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    train_df.to_csv(META_DIR / "train_pairs.csv", index=False)
    val_df.to_csv(META_DIR / "val_pairs.csv", index=False)
    test_df.to_csv(META_DIR / "test_pairs.csv", index=False)

    print("\nSUCCESS ✅")
    print(f"Train pairs: {len(train_df)}")
    print(f"Val pairs:   {len(val_df)}")
    print(f"Test pairs:  {len(test_df)}")


if __name__ == "__main__":
    main()