import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import wandb
from config import CONFIG
from ddpm import DDPM


class utilities:
    @staticmethod
    def save_checkpoint(model, optimizer, scheduler, epoch, run):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "T": model.T,  # save hyperparameters needed to reconstruct the model
            "config": {
                "BATCH_SIZE": run.config.BATCH_SIZE,
                "IMAGE_SIZE": run.config.IMAGE_SIZE,
                "NUM_EPOCHS": run.config.NUM_EPOCHS,
                "MIN_LR": run.config.MIN_LR,
                "MAX_LR": run.config.MAX_LR,
                "WARMUP_STEPS": run.config.WARMUP_STEPS,
                "WEIGHT_DECAY": run.config.WEIGHT_DECAY,
                "DATASET_PATH": run.config.DATASET_PATH,
                "MODELS_PATH": run.config.MODELS_PATH,
            },
        }

        # save locally first
        checkpoint_path = f"{run.config.MODELS_PATH}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # log to wandb as an artifact
        artifact = wandb.Artifact(
            name="ddpm-checkpoint",
            type="model",
            description=f"DDPM checkpoint at epoch {epoch}",
        )
        artifact.add_file(checkpoint_path)
        run.log_artifact(artifact)

        print(f"Checkpoint saved at epoch {epoch}")

    @staticmethod
    def load_checkpoint(run, artifact_name, device):
        # download artifact from wandb
        artifact = run.use_artifact(
            f"{artifact_name}:latest"
        )  # or specify version like :v0
        artifact_dir = artifact.download()

        # find the checkpoint file
        checkpoint_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # reconstruct model
        model = DDPM(T=checkpoint["T"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        return model, checkpoint

    @staticmethod
    def infer(model, run, num_infer_imgs=5):
        with torch.no_grad():
            for i in range(num_infer_imgs):
                run.log(
                    {
                        "Images Generated": wandb.Image(
                            DDPM.denorm(model.reverse_process(run.config))[0],
                            caption=f"Image {i + 5}",
                        )
                    }
                )
