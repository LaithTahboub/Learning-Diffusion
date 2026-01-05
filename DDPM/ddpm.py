# Author: Laith Tahboub
# Date: Jan 4, 2026

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
from unet import UNet


class DDPM(nn.Module):
    alpha_bar: torch.Tensor
    beta: torch.Tensor
    alpha: torch.Tensor

    def __init__(self, beta_start=1e-4, beta_end=0.02, T=1000):
        super().__init__()
        self.T = T

        self.register_buffer("beta", torch.linspace(beta_start, beta_end, T))
        self.register_buffer("alpha", 1.0 - self.beta)
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))

        self.unet = UNet(T)
        self.loss_fn = nn.MSELoss()

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor):
        epsilon = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)

        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(
            1.0 - alpha_bar_t
        ) * epsilon, epsilon

    # @torch.no_grad()
    def reverse_process(self, config):
        device = self.alpha.device
        batch_size = config.BATCH_SIZE
        image_size = torch.Size([3, config.IMAGE_SIZE, config.IMAGE_SIZE])

        x_t = torch.randn(
            batch_size, image_size[0], image_size[1], image_size[2], device=device
        )

        for t in range(self.T - 1, 0, -1):
            if t % 100 == 0:
                print(f"Step {t}")

            t_batched = torch.full((batch_size,), t, device=device, dtype=torch.long)
            eps_theta = self.unet(x_t, t_batched)  # (B, C, H, W)

            # make math vars batched
            alpha_t = self.alpha[t].expand(batch_size).view(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_bar[t].expand(batch_size).view(-1, 1, 1, 1)
            beta_t = self.beta[t].expand(batch_size).view(-1, 1, 1, 1)

            mean = (1 / torch.sqrt(alpha_t)) * (
                x_t - (1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t)) * eps_theta
            )
            std = torch.sqrt(beta_t)

            if t > 0:
                z = torch.randn_like(x_t)
                x_t = mean + std * z
            else:
                x_t = mean

        return x_t

    @staticmethod
    def denorm(x: torch.Tensor):
        return ((x + 1) / 2).clamp(0, 1)
