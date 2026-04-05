import torch
import torch.nn.functional as F


# -------------------------------
# STFT magnitude (for spectral loss)
# -------------------------------
def stft_mag(x, n_fft=512, hop_length=128, win_length=512):
    """
    x: (B, 1, T)
    returns: (B, F, time)
    """
    window = torch.hann_window(win_length).to(x.device)

    spec = torch.stft(
        x.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )

    return torch.abs(spec)


# -------------------------------
# Reconstruction Loss (L1)
# -------------------------------
def recon_loss(enhanced, clean):
    return F.l1_loss(enhanced, clean)


# -------------------------------
# Spectral Loss (important for audio quality)
# -------------------------------
def spectral_loss(enhanced, clean):
    return F.l1_loss(stft_mag(enhanced), stft_mag(clean))


# -------------------------------
# Smoothness Loss (reduces noise spikes)
# -------------------------------
def smoothness_loss(enhanced):
    diff = enhanced[:, :, 1:] - enhanced[:, :, :-1]
    return torch.mean(torch.abs(diff))


# -------------------------------
# Sparsity Loss (optional)
# -------------------------------
def sparsity_loss(enhanced):
    return torch.mean(torch.abs(enhanced))


# -------------------------------
# Total Loss with lambda weights
# -------------------------------
def total_loss(enhanced, clean, lambdas):
    """
    lambdas = {
        "recon": 1.0,
        "spec": 1.0,
        "smooth": 0.1,
        "sparse": 0.0
    }
    """

    l_rec = recon_loss(enhanced, clean)
    l_spec = spectral_loss(enhanced, clean)
    l_smooth = smoothness_loss(enhanced)
    l_sparse = sparsity_loss(enhanced)

    total = (
        lambdas["recon"] * l_rec +
        lambdas["spec"] * l_spec +
        lambdas["smooth"] * l_smooth +
        lambdas["sparse"] * l_sparse
    )

    return total, {
        "recon": l_rec.item(),
        "spec": l_spec.item(),
        "smooth": l_smooth.item(),
        "sparse": l_sparse.item(),
        "total": total.item()
    }