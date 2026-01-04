import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, t_emb=None):
        x = self.conv1(x)
        if t_emb is not None:
            x = x + t_emb
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, T):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Embedding(T, 128),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.time_projs = nn.ModuleList(
            [
                nn.Linear(128, 64),
                nn.Linear(128, 128),
                nn.Linear(128, 256),
                nn.Linear(128, 512),
                nn.Linear(128, 1024),
            ]
        )

        self.down1 = ConvBlock(3, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.down4 = ConvBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.up4 = ConvBlock(1024, 512)
        self.up3 = ConvBlock(512, 256)
        self.up2 = ConvBlock(256, 128)
        self.up1 = ConvBlock(128, 64)

        self.pool = nn.MaxPool2d(2, 2)

        self.convT4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.convT3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.convT2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.convT1 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.last_conv = nn.Conv2d(64, 3, 1)

    def forward(self, image, t):
        t_emb = self.time_mlp(t)
        t64, t128, t256, t512, t1024 = [
            proj(t_emb)[:, :, None, None] for proj in self.time_projs
        ]

        d1 = self.down1(image, t64)
        d2 = self.down2(self.pool(d1), t128)
        d3 = self.down3(self.pool(d2), t256)
        d4 = self.down4(self.pool(d3), t512)

        b = self.bottleneck(self.pool(d4), t1024)

        u4 = self.up4(self.skip(d4, self.convT4(b)), t512)
        u3 = self.up3(self.skip(d3, self.convT3(u4)), t256)
        u2 = self.up2(self.skip(d2, self.convT2(u3)), t128)
        u1 = self.up1(self.skip(d1, self.convT1(u2)), t64)

        return self.last_conv(u1)

    def skip(self, enc, dec):  # handle size mismatch when concating by interpolation
        if enc.shape[2:] != dec.shape[2:]:
            dec = F.interpolate(
                dec, size=enc.shape[2:], mode="bilinear", align_corners=False
            )
        return torch.cat([enc, dec], dim=1)
