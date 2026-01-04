import argparse

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from config import *
from ddpm import DDPM
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
from unet import UNet

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
            # # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

    train_size = int(len(dataset) * 0.8) - (int(len(dataset) * 0.8) % batch_size)
    print(len(dataset))
    print(train_size)
    val_size = int(len(dataset) * ((len(dataset) - train_size) / len(dataset)) / 2)
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # datas[n_images, C, H, W]

    return train_loader, val_loader, test_loader


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
    print(img_size)
    print(batch_size)

    # init data, model, and optimizer
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_data, val_data, test_data = load_data(
        img_size=img_size, batch_size=batch_size
    )

    model = DDPM(T=T)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # train process

    for epoch in range(min(50, epochs)):
        epoch_loss = 0.0
        num_batches = 0

        for b, batch in enumerate(train_data):
            images, _ = batch
            images = images.to(device)

            rand_timestamps = torch.randint(
                0, 999, torch.Size([batch_size]), device=device
            )

            noisy_images, epsilon = model.forward_process(images, rand_timestamps)

            batch_preds = model.unet.forward(noisy_images, rand_timestamps)

            loss = model.loss_fn(epsilon, batch_preds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # for x, img in enumerate(images):
            #     save_image(
            #         img,
            #         f"/fs/nexus-scratch/ltahboub/learning-diffusion/DDPM/del/img{x}.png",
            #     )
            #     if x > 10:
            #         break

            # break

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / num_batches:.4f}")

    # test with some basic inference

    torch.cuda.empty_cache()
    with torch.no_grad():
        save_image(model.reverse_process(1), "basic-test2.png")


if __name__ == "__main__":
    main()
