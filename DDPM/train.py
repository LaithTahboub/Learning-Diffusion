import argparse
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from config import CONFIG
from ddpm import DDPM
from PIL import Image
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from utilities import utilities

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
#


def load_data(config):
    transform = transforms.Compose(
        [
            transforms.Resize(config.IMAGE_SIZE),
            # # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset = datasets.ImageFolder(config.DATASET_PATH, transform=transform)

    train_size = int(len(dataset) * 0.8) - (int(len(dataset) * 0.8) % config.BATCH_SIZE)
    print(len(dataset))
    print(train_size)
    val_size = int(len(dataset) * ((len(dataset) - train_size) / len(dataset)) / 2)
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # datas[n_images, C, H, W]

    return train_loader, val_loader, test_loader


def main():
    with wandb.init(project="DDPM", config=CONFIG, name="Big Test") as run:
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_epochs", type=int, default=run.config.NUM_EPOCHS)
        parser.add_argument("--T", type=int, default=run.config.T_)
        parser.add_argument("--batch_size", type=int, default=run.config.BATCH_SIZE)
        parser.add_argument("--num_images", type=int, default=run.config.DATASET_SIZE)
        parser.add_argument("--img_size", type=int, default=run.config.IMAGE_SIZE)
        parser.add_argument("--min_lr", type=float, default=run.config.MIN_LR)
        parser.add_argument("--max_lr", type=float, default=run.config.MAX_LR)
        parser.add_argument("--warmup_steps", type=int, default=run.config.WARMUP_STEPS)
        parser.add_argument(
            "--weight_decay", type=float, default=run.config.WEIGHT_DECAY
        )
        parser.add_argument(
            "--resume",
            type=str,
            default=None,
            help="wandb artifact name to resume from",
        )

        # other args

        # initialize args
        args = parser.parse_args()

        # hyperparameters
        num_epochs = args.num_epochs
        T = args.T
        # batch_size = args.batch_size
        # num_images = args.num_images
        # img_size = args.img_size
        # weight_decay = args.weight_decay
        # warmup_steps = args.warmup_steps
        # min_lr = args.min_lr
        # max_lr = args.max_lr

        # print(img_size)
        # print(batch_size)

        # init data, model, and optimizer
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        train_data, val_data, test_data = load_data(run.config)

        start_epoch = 0

        if args.resume:
            # Resume from checkpoint
            model, checkpoint = utilities.load_checkpoint(run, args.resume, device)

            optimizer = AdamW(
                params=model.parameters(),
                lr=run.config.MAX_LR,
                weight_decay=run.config.WEIGHT_DECAY,
            )
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Recreate scheduler and load state
            scheduler = lr_scheduler.SequentialLR(
                optimizer=optimizer,
                schedulers=[
                    lr_scheduler.LambdaLR(
                        optimizer, lr_lambda=lambda step: step / run.config.WARMUP_STEPS
                    ),
                    lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=num_epochs * len(train_data) - run.config.WARMUP_STEPS,
                        eta_min=run.config.MIN_LR,
                    ),
                ],
                milestones=[run.config.WARMUP_STEPS],
            )
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed from epoch {start_epoch}")
        else:
            # Fresh start
            model = DDPM(T=T)
            model.to(device)

            optimizer = AdamW(
                params=model.parameters(),
                lr=run.config.MAX_LR,
                weight_decay=run.config.WEIGHT_DECAY,
            )

            scheduler = lr_scheduler.SequentialLR(
                optimizer=optimizer,
                schedulers=[
                    lr_scheduler.LambdaLR(
                        optimizer, lr_lambda=lambda step: step / run.config.WARMUP_STEPS
                    ),
                    lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=num_epochs * len(train_data) - run.config.WARMUP_STEPS,
                        eta_min=run.config.MIN_LR,
                    ),
                ],
                milestones=[run.config.WARMUP_STEPS],
            )

        run.watch(model, log="all", log_freq=10)

        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for b, batch in enumerate(train_data):
                images, _ = batch
                images = images.to(device)

                rand_timestamps = torch.randint(0, T, (images.shape[0],), device=device)

                noisy_images, epsilon = model.forward_process(images, rand_timestamps)

                batch_preds = model.unet.forward(noisy_images, rand_timestamps)

                loss = model.loss_fn(epsilon, batch_preds)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                run.log(
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

            avg_loss = epoch_loss / num_batches
            scheduler.step()

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            if (epoch + 1) % 100 == 0:
                utilities.save_checkpoint(model, optimizer, scheduler, epoch, run)
            if (epoch + 1) % 300 == 0:
                utilities.infer(model, run, num_infer_imgs=1)

        # save final checkpoint
        utilities.save_checkpoint(model, optimizer, scheduler, num_epochs - 1, run)

        # test with some basic inference

        torch.cuda.empty_cache()

        utilities.infer(model, run, num_infer_imgs=5)

        run.finish()


if __name__ == "__main__":
    main()
