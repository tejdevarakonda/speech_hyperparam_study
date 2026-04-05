# Speech Enhancement Hyperparameter Study

A deep learning-based speech enhancement system using U-Net architecture to remove background noise from speech signals. This project explores hyperparameter optimization for training speech enhancement models on the VoiceBank dataset.

## Project Overview

This repository contains a complete pipeline for training and evaluating speech enhancement models. The system uses a 1D U-Net convolutional architecture to map noisy speech to clean speech, with a multi-component loss function combining reconstruction, spectral, smoothness, and sparsity terms.

### Key Features
- **U-Net Architecture**: 1D convolutional encoder-decoder for speech enhancement
- **Multi-component Loss Function**: Combines reconstruction, spectral, smoothness, and sparsity losses
- **Comprehensive Evaluation**: SI-SDR and STOI metrics for audio quality assessment
- **Hyperparameter Optimization**: Study and track different training configurations
- **Data Pipeline**: Preprocessing and augmentation for clean/noisy speech pairs

## Dataset

The project uses the **VoiceBank dataset** with:
- **28 speakers** in the training set
- **Clean and noisy speech pairs** for supervised learning
- **Train/validation/test splits** with metadata in CSV format
- **Sample rate**: 16 kHz
- **Segment length**: 2.0 seconds (configurable)

### Data Structure
```
data/
├── raw/                          # Original audio files
│   ├── clean_trainset_28spk_wav/
│   ├── noisy_trainset_28spk_wav/
│   ├── clean_testset_wav/
│   └── noisy_testset_wav/
├── processed/                    # Preprocessed audio
│   ├── train/ (clean/, noisy/)
│   ├── val/ (clean/, noisy/)
│   └── test/ (clean/, noisy/)
└── metadata/                     # CSV files with audio pairing info
    ├── train_pairs.csv
    ├── val_pairs.csv
    └── test_pairs.csv
```

## Project Structure

```
speech_hyperparam_study/
├── src/                          # Source code
│   ├── model.py                  # U-Net model architecture
│   ├── train.py                  # Training loop and utilities
│   ├── evaluate.py               # Evaluation metrics and inference
│   ├── dataset.py                # VoiceBankDataset class
│   ├── losses.py                 # Loss function definitions
│   ├── utils.py                  # Utility functions
│   └── __init__.py
├── scripts/                      # Data preparation scripts
│   ├── create_folders.py         # Create directory structure
│   ├── create_pairs.py           # Generate CSV pairs from audio files
│   └── preprocess.py             # Preprocess and save audio
├── data/                         # Dataset directory
├── checkpoints/                  # Trained model weights
├── logs/                         # Training history and logs
├── outputs/                      # Evaluation results and enhanced audio
└── README.md
```

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU support)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd speech_hyperparam_study
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- torch
- numpy
- pandas
- tqdm
- pystoi (for STOI metric)
- scipy

## Usage

### 1. Data Preparation

First, ensure raw audio files are in the `data/raw/` directory, then:

```bash
# Create folder structure
python scripts/create_folders.py

# Generate audio pair CSV files
python scripts/create_pairs.py

# Preprocess and save audio files
python scripts/preprocess.py
```

### 2. Training

Train a model with specific hyperparameters:

```bash
python src/train.py \
    --train_csv data/metadata/train_processed_pairs.csv \
    --val_csv data/metadata/val_processed_pairs.csv \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --segment_seconds 2.0 \
    --lambda_recon 1.0 \
    --lambda_spec 1.0 \
    --lambda_smooth 0.01 \
    --lambda_sparse 0.01 \
    --checkpoint_dir checkpoints/exp_01
```

#### Training Hyperparameters
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--segment_seconds`: Audio segment length in seconds (default: 2.0)
- `--lambda_recon`: Weight for reconstruction loss (default: 1.0)
- `--lambda_spec`: Weight for spectral loss (default: 1.0)
- `--lambda_smooth`: Weight for smoothness loss (default: 0.01)
- `--lambda_sparse`: Weight for sparsity loss (default: 0.01)
- `--checkpoint_dir`: Directory to save checkpoints (default: checkpoints/exp_01)

### 3. Evaluation

Evaluate a trained model on the test set:

```bash
python src/evaluate.py \
    --test_csv data/metadata/test_processed_pairs.csv \
    --checkpoint checkpoints/exp_01/best_model.pt \
    --out_dir outputs \
    --segment_seconds 2.0 \
    --batch_size 1
```

#### Evaluation Output
- **test_metrics.csv**: SI-SDR and STOI metrics for each test sample
- **enhanced_audio/**: Enhanced speech waveforms saved as WAV files

### 4. Model Architecture

The model uses a 1D U-Net architecture:
- **Input**: Noisy speech waveform (batch_size, 1, segment_length)
- **Encoding**: Progressive downsampling with convolutional blocks
- **Bottleneck**: Feature extraction at the lowest resolution
- **Decoding**: Progressive upsampling with skip connections
- **Output**: Enhanced speech waveform (batch_size, 1, segment_length)

Key components:
- Conv blocks with BatchNorm and ReLU activation
- Skip connections between encoder and decoder
- 1D convolutions for temporal operations

## Loss Function

The total loss is a weighted combination of four components:

$$\text{Loss} = \lambda_{\text{recon}} \cdot L_{\text{recon}} + \lambda_{\text{spec}} \cdot L_{\text{spec}} + \lambda_{\text{smooth}} \cdot L_{\text{smooth}} + \lambda_{\text{sparse}} \cdot L_{\text{sparse}}$$

- **Reconstruction Loss (L1)**: Direct waveform difference
- **Spectral Loss (L1)**: STFT magnitude difference for frequency-domain accuracy
- **Smoothness Loss**: Temporal smoothness regularization
- **Sparsity Loss**: Sparsity-inducing regularization

## Evaluation Metrics

- **SI-SDR (Scale-Invariant Signal-to-Noise Ratio)**: Measures speech quality preservation, scale-invariant
- **STOI (Short-Time Objective Intelligibility)**: Measures speech intelligibility (0-1 scale)

## Results

Trained models and their configurations are stored in:
- `checkpoints/exp_01/best_model.pt`: Best model weights
- `checkpoints/exp_01/config.json`: Training configuration
- `logs/exp_01/history.csv`: Training history (loss, metrics per epoch)

## Hyperparameter Study

This project supports experiments with different hyperparameter combinations:

Example configurations to explore:
- Loss function weights (lambda values)
- Learning rates and schedulers
- Batch sizes and segment lengths
- Different network architectures (depth, channels)
- Optimizer selections

Results are organized by experiment (exp_01, exp_02, etc.) in corresponding directories.

## Development

### Project Dependencies Overview
- **PyTorch**: Deep learning framework
- **NumPy/SciPy**: Numerical operations and signal processing
- **Pandas**: CSV data handling
- **pystoi**: STOI metric calculation
- **tqdm**: Progress bars

### Code Style
- Python 3.8+ features
- Type hints recommended
- Modular function design

## License

[Add your license here]

## Citation

If you use this project, please cite:

```
[Add citation information]
```

## Contact

For questions or issues, please open an issue on the repository.

---

**Last Updated**: April 2026
