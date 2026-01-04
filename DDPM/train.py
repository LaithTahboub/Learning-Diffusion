import argparse

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ddpm import DDPM
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
from unet import UNet
from config import *

# Train algo:
# repeat:
#   x_0 ~ q(x_0) -- choose some random sample from the data
#   t ~ Uniform({1, ..., T}) -- choose some random timestamp uniformly, based on chosen T
#   ε ~ N(0, I) -- create some noise with the same size of the data samples
#   Take gradient descent step on
#       delta_theta ||ε - ε_theta (sqrt(alpha-bar_t) x_0 + sqrt(1 - alpha-bar_t) * ε , t)||^ 2
#   until converged
# for (int i = 0; i < 100; )
#

def load_data(batch_size=32, img_size=64):
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(
        DATASET_PATH, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--T", type=int, default=T_)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_images", type=int, default=DATASET_SIZE)
    parser.add_argument("--img_size", type=int, default=IMAGE_SIZE)

    # other args

    # initialize args
    args = parser.parse_args()

    # hyperparameters
    epochs = args.epochs
    T = args.T
    batch_size = args.batch_size
    num_images = args.num_images
    img_size = args.img_size


    # init data and model

    data = load_data()

    # train process
    model = DDPM(T = T)

    for epoch in range(epochs):
        for batch in range(batch_size // num_images)
            
            
if __name__ == "__main__":
    main()
