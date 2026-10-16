"""Residual Encoder 3D U-Net (nnUNetResEncUNet).

Conforms to nnU-Net v2 Residual Encoder architectural specifications:
- Residual basic blocks in encoder stages
- Instance normalization and LeakyReLU activations
- Strided convolution downsampling
- Transposed convolution upsampling with skip concatenation
- Final 3-channel projection (TC, WT, ET) for BraTS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """3D Residual Basic Block with InstanceNorm and LeakyReLU."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.act2(out + residual)
        return out


class DecoderBlock3D(nn.Module):
    """3D Decoder Block with Transposed Conv Upsampling and Conv refinement."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.norm_up = nn.InstanceNorm3d(out_channels, affine=True)
        self.act_up = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x, skip):
        x = self.act_up(self.norm_up(self.up(x)))
        # Pad if odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            diff = [s - x_s for s, x_s in zip(skip.shape[2:], x.shape[2:])]
            x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResEncUNet(nn.Module):
    """Residual Encoder 3D U-Net matching nnU-Net resenc presets (M, L, XL).
    
    Presets:
    - M:  channels = (32, 64, 128, 256)
    - L:  channels = (48, 96, 192, 384)
    - XL: channels = (64, 128, 256, 512)
    """
    PRESETS = {
        'M': (32, 64, 128, 256),
        'L': (48, 96, 192, 384),
        'XL': (64, 128, 256, 512),
    }

    def __init__(self, in_channels=4, out_channels=3, preset='M', channels=None):
        super().__init__()
        if channels is None:
            channels = self.PRESETS.get(preset.upper(), self.PRESETS['M'])
        self.channels = channels
        c0, c1, c2, c3 = channels

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c0, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(c0, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        # Residual Encoder stages
        self.enc0 = ResidualBlock3D(c0, c0, stride=1)
        self.enc1 = ResidualBlock3D(c0, c1, stride=2)
        self.enc2 = ResidualBlock3D(c1, c2, stride=2)
        self.enc3 = ResidualBlock3D(c2, c3, stride=2)

        # Bottleneck
        self.bottleneck = ResidualBlock3D(c3, c3, stride=1)

        # Decoder stages
        self.dec2 = DecoderBlock3D(in_channels=c3, skip_channels=c2, out_channels=c2)
        self.dec1 = DecoderBlock3D(in_channels=c2, skip_channels=c1, out_channels=c1)
        self.dec0 = DecoderBlock3D(in_channels=c1, skip_channels=c0, out_channels=c0)

        # Final projection head
        self.head = nn.Conv3d(c0, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        s0 = self.stem(x)
        e0 = self.enc0(s0)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder
        d2 = self.dec2(b, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)

        # Output logits
        return self.head(d0)


def build_resenc_unet(preset='M', in_channels=4, out_channels=3, weights_path=None, device='cpu'):
    """Instantiate and optionally load weights for ResEncUNet."""
    model = ResEncUNet(in_channels=in_channels, out_channels=out_channels, preset=preset)
    if weights_path is not None:
        p = Path(weights_path)
        if p.exists():
            state = torch.load(p, map_location=device, weights_only=False)
            sd = state.get('state_dict', state.get('model', state))
            model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model
