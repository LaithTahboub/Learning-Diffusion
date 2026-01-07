import os
from argparse import ArgumentParser

import torch
from diffusers import DDPMScheduler, DiffusionPipeline
from PIL import Image


class CoupledDiffusion(DiffusionPipeline):
    def __init__(
        self,
        unet_a,
        unet_b,
        scheduler_a,
        scheduler_b,
        text_encoder,
        tokenizer,
        vae=None,
    ):
        super().__init__()
        self.register_modules(
            unet_a=unet_a,
            unet_b=unet_b,
            scheduler_a=scheduler_a,
            scheduler_b=scheduler_b,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
        )

    # given some text, uses the class' encoder to first tokenize and then encode prompt
    def encode_prompt(self, prompt: str, device):
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        return self.text_encoder(tokens)[0]

    # coupled diffusion sampling:
    @torch.no_grad()
    def __call__(
        self,
        prompt_a: str,
        prompt_b: str,
        lambda_: float,
        batch_size: int = 1,
        num_inference_steps: int = 75,
        guidance_scale: float = 7.5,
        seed: int = 32,
        latents: torch.Tensor | None = None,
    ):
        device = self.unet_a.device
        dtype = self.unet_a.dtype

        # get the two prompts encoded and the unconditional prompt ""
        cond_a = self.encode_prompt(prompt_a, device=device)
        cond_b = self.encode_prompt(prompt_b, device=device)
        uncond = self.encode_prompt("", device=device)

        # make some noise :D. both latents start at same noise
        if latents is None:
            latents = torch.randn(
                (
                    batch_size,
                    self.unet_a.config.in_channels,
                    self.unet_a.config.sample_size,
                    self.unet_a.config.sample_size,
                ),
                device=device,
                dtype=dtype,
            )

        latents_a = latents.clone()
        latents_b = latents.clone()

        # set the timestamps for the schedulers to T
        self.scheduler_a.set_timesteps(num_inference_steps, device=device)
        self.scheduler_b.set_timesteps(num_inference_steps, device=device)

        # set up generators. try both same and different seeds for testing:
        gen_a = torch.Generator(device=device)
        gen_b = torch.Generator(device=device)
        gen_a.manual_seed(seed)
        gen_b.manual_seed(seed)

        def cfg_eps(unet, latents_x, t_scalar, cond):
            eps_u = unet(latents_x, t_scalar, encoder_hidden_states=uncond).sample
            eps_c = unet(latents_x, t_scalar, encoder_hidden_states=cond).sample
            return eps_u + guidance_scale * (eps_c - eps_u)

        timesteps = self.scheduler_a.timesteps

        for i, t in enumerate(timesteps):
            t_idx = int(t.item())

            # get eps with applying classifier free guidance
            eps_a = cfg_eps(self.unet_a, latents_a, t, cond_a)
            eps_b = cfg_eps(self.unet_b, latents_b, t, cond_b)

            # get the x0 predictions
            alpha_bar_t = self.scheduler_a.alphas_cumprod[t_idx].to(
                device=device, dtype=dtype
            )
            sqrt_alpha = alpha_bar_t.sqrt()
            sqrt_one_minus_alpha = (1.0 - alpha_bar_t).sqrt()

            x0_a = (latents_a - sqrt_one_minus_alpha * eps_a) / sqrt_alpha
            x0_b = (latents_b - sqrt_one_minus_alpha * eps_b) / sqrt_alpha

            # ---- proper DDPM stochastic step (scheduler handles noise scale) ----
            out_a = self.scheduler_a.step(eps_a, t, latents_a, generator=gen_a)
            out_b = self.scheduler_b.step(eps_b, t, latents_b, generator=gen_b)

            latents_a = out_a.prev_sample
            latents_b = out_b.prev_sample

            # ---- coupling (Algorithm 1 style, using latent x0 approx) ----
            # scale = sqrt(1 - alpha_bar_{t-1})
            if i < len(timesteps) - 1:
                t_prev_int = int(timesteps[i + 1].item())
                alpha_prev = self.scheduler_a.alphas_cumprod[t_prev_int].to(
                    device=device, dtype=dtype
                )
                scale = (1.0 - alpha_prev).sqrt()
            else:
                scale = torch.tensor(0.0, device=device, dtype=dtype)

            if lambda_ != 0.0:
                latents_a = latents_a - scale * lambda_ * (x0_a - x0_b)
                latents_b = latents_b - scale * lambda_ * (x0_b - x0_a)

        return latents_a, latents_b

    @staticmethod
    def retrieve_sd_pipeline(model="manojb/stable-diffusion-2-1-base"):
        pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
        pipe.to("cuda")
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        return pipe


def main():
    parser = ArgumentParser()
    parser.add_argument("--prompt_a", type=str, required=True)
    parser.add_argument("--prompt_b", type=str, required=True)
    parser.add_argument("--lambda_", type=float, default=0.05)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/fs/nexus-scratch/ltahboub/learning-diffusion/CoupledDiffusion/inference",
    )
    args = parser.parse_args()

    pipe = CoupledDiffusion.retrieve_sd_pipeline()

    cd = CoupledDiffusion(
        unet_a=pipe.unet,
        unet_b=pipe.unet,  # same weights is fine
        scheduler_a=pipe.scheduler,
        scheduler_b=pipe.scheduler,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        vae=pipe.vae,
    )

    latents_a, latents_b = cd(
        args.prompt_a,
        args.prompt_b,
        args.lambda_,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    # ---- decode latents -> images ----
    with torch.no_grad():
        images_a = pipe.vae.decode(latents_a / pipe.vae.config.scaling_factor).sample
        images_b = pipe.vae.decode(latents_b / pipe.vae.config.scaling_factor).sample

    def to_pil(batch):
        batch = (batch / 2 + 0.5).clamp(0, 1)
        batch = (batch * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
        return [Image.fromarray(img) for img in batch]

    os.makedirs(args.output_dir, exist_ok=True)
    for i, img in enumerate(to_pil(images_a)):
        img.save(f"{args.output_dir}/a_{i}.png")
    for i, img in enumerate(to_pil(images_b)):
        img.save(f"{args.output_dir}/b_{i}.png")


if __name__ == "__main__":
    main()
