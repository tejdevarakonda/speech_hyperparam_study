from pathlib import Path

folders = [
    "data/processed/train/clean",
    "data/processed/train/noisy",
    "data/processed/val/clean",
    "data/processed/val/noisy",
    "data/processed/test/clean",
    "data/processed/test/noisy",
    "data/metadata",
    "checkpoints",
    "logs",
    "outputs",
    "scripts",
    "src",
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

print("Folders created successfully.")