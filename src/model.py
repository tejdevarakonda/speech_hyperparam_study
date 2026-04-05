import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# Basic convolution block
# -------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------
# Downsampling block
# -------------------------------
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------
# Upsampling block
# -------------------------------
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1
        )
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # Fix size mismatch (very important)
        if x.shape[-1] > skip.shape[-1]:
            x = x[..., :skip.shape[-1]]
        elif x.shape[-1] < skip.shape[-1]:
            pad = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (0, pad))

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# -------------------------------
# Full U-Net Model
# -------------------------------
class SpeechEnhancementUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(1, 16)
        self.down1 = Down(16, 32)

        self.enc2 = ConvBlock(32, 32)
        self.down2 = Down(32, 64)

        self.enc3 = ConvBlock(64, 64)
        self.down3 = Down(64, 128)

        self.enc4 = ConvBlock(128, 128)
        self.down4 = Down(128, 256)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 256)

        # Decoder
        self.up4 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up1 = Up(32, 16)

        # Output layer
        self.final = nn.Conv1d(16, 1, kernel_size=1)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        x = self.down1(s1)

        s2 = self.enc2(x)
        x = self.down2(s2)

        s3 = self.enc3(x)
        x = self.down3(s3)

        s4 = self.enc4(x)
        x = self.down4(s4)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        # Output
        x = self.final(x)
        return self.output_activation(x)