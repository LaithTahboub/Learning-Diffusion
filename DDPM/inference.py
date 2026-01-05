import torch
import wandb
from ddpm import DDPM
from torchvision.utils import save_image
import os

def inference(artifact_name, num_images=5, image_size=(3, 64, 64)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb just to download the artifact
    with wandb.init(project="DDPM", job_type="inference") as run:
        artifact = run.use_artifact(f"{artifact_name}:latest")
        artifact_dir = artifact.download()

        checkpoint_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = DDPM(T=checkpoint["T"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        with torch.no_grad():
            for i in range(num_images):
                img = model.reverse_process(run.config)
                img = DDPM.denorm(img)
                save_image(img, f"generated_{i}.png")
                print(f"Saved generated_{i}.png")

                print(f"Saved generated_{i}.png")

if __name__ == "__main__":
    # inference("your-wandb-username/DDPM/ddpm-checkpoint")
