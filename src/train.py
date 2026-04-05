import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VoiceBankDataset
from model import SpeechEnhancementUNet
from losses import total_loss

def run_epoch(model, loader, optimizer, device, lambdas, train=True):
    if train:
        model.train()
    else:
        model.eval()

    totals = {
        "recon": 0.0,
        "spec": 0.0,
        "smooth": 0.0,
        "sparse": 0.0,
        "total": 0.0,
    }

    num_batches = 0

    for noisy, clean in tqdm(loader, leave=False):
        noisy = noisy.to(device)
        clean = clean.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            enhanced = model(noisy)
            loss, parts = total_loss(enhanced, clean, lambdas)

            if train:
                loss.backward()
                optimizer.step()

        for k in totals:
            totals[k] += parts[k]
        num_batches += 1

    for k in totals:
        totals[k] /= max(num_batches, 1)

    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="data/metadata/train_processed_pairs.csv")
    parser.add_argument("--val_csv", type=str, default="data/metadata/val_processed_pairs.csv")
    parser.add_argument("--exp_name", type=str, default="exp_01")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--segment_seconds", type=float, default=2.0)

    parser.add_argument("--recon_w", type=float, default=1.0)
    parser.add_argument("--spec_w", type=float, default=1.0)
    parser.add_argument("--smooth_w", type=float, default=0.1)
    parser.add_argument("--sparse_w", type=float, default=0.0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt_dir = Path("checkpoints") / args.exp_name
    log_dir = Path("logs") / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    train_ds = VoiceBankDataset(
        args.train_csv,
        segment_seconds=args.segment_seconds,
        train=True,
    )
    val_ds = VoiceBankDataset(
        args.val_csv,
        segment_seconds=args.segment_seconds,
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = SpeechEnhancementUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lambdas = {
        "recon": args.recon_w,
        "spec": args.spec_w,
        "smooth": args.smooth_w,
        "sparse": args.sparse_w,
    }

    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, lambdas, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, lambdas, train=False)

        row = {
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "val_total": val_metrics["total"],
            "train_recon": train_metrics["recon"],
            "val_recon": val_metrics["recon"],
            "train_spec": train_metrics["spec"],
            "val_spec": val_metrics["spec"],
            "train_smooth": train_metrics["smooth"],
            "val_smooth": val_metrics["smooth"],
            "train_sparse": train_metrics["sparse"],
            "val_sparse": val_metrics["sparse"],
        }
        history.append(row)

        pd.DataFrame(history).to_csv(log_dir / "history.csv", index=False)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train {train_metrics['total']:.4f} | "
            f"Val {val_metrics['total']:.4f}"
        )

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val": best_val,
                    "config": config,
                },
                ckpt_dir / "best_model.pt",
            )

    print(f"Training finished. Best validation loss: {best_val:.4f}")
    print("Saved checkpoint:", ckpt_dir / "best_model.pt")


if __name__ == "__main__":
    main()