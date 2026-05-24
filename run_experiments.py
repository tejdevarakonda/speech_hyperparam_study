import itertools
import subprocess
import sys
from pathlib import Path

import pandas as pd


# -----------------------------
# Hyperparameter grid
# -----------------------------
LAMBDA1_VALUES = [0.01, 0.1, 1, 10]   # spec_w
LAMBDA2_VALUES = [0.01, 0.1, 1, 10]   # smooth_w

RECON_W = 1.0
SPARSE_W = 0.0

# Change this if you want longer runs
EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-3
SEGMENT_SECONDS = 2.0

SUMMARY_DIR = Path("outputs")
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = SUMMARY_DIR / "grid_search_summary.csv"


def run_command(cmd):
    print("\nRunning:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main():
    results = []
    combos = list(itertools.product(LAMBDA1_VALUES, LAMBDA2_VALUES))

    for i, (lambda1, lambda2) in enumerate(combos, start=1):
        exp_name = f"exp_l1_{lambda1}_l2_{lambda2}"

        ckpt_dir = Path("checkpoints") / exp_name
        out_dir = Path("outputs") / exp_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable, "src/train.py",
            "--exp_name", exp_name,
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--lr", str(LR),
            "--segment_seconds", str(SEGMENT_SECONDS),
            "--recon_w", str(RECON_W),
            "--spec_w", str(lambda1),
            "--smooth_w", str(lambda2),
            "--sparse_w", str(SPARSE_W),
        ]
        run_command(train_cmd)

        eval_cmd = [
            sys.executable, "src/evaluate.py",
            "--checkpoint", str(ckpt_dir / "best_model.pt"),
            "--out_dir", str(out_dir),
            "--segment_seconds", str(SEGMENT_SECONDS),
        ]
        run_command(eval_cmd)

        metrics_file = out_dir / "test_metrics.csv"
        df = pd.read_csv(metrics_file)

        row = {
            "exp_name": exp_name,
            "lambda1_spec": lambda1,
            "lambda2_smooth": lambda2,
            "recon_w": RECON_W,
            "sparse_w": SPARSE_W,
            "epochs": EPOCHS,
            "mean_stoi": df["stoi"].mean(),
            "mean_si_sdr": df["si_sdr"].mean(),
        }
        results.append(row)

        pd.DataFrame(results).to_csv(SUMMARY_FILE, index=False)
        print(f"\nSaved summary after {i}/{len(combos)} runs -> {SUMMARY_FILE}")

    print("\nAll experiments completed successfully.")
    print(f"Final summary saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()