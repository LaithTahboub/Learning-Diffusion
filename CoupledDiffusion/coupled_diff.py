import os
from argparse import ArgumentParser
from types import SimpleNamespace

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image


class CoupledDiffusion:
    def __init__(self, pipe_a: DiffusionPipeline, pipe_b: DiffusionPipeline):
        self.pipe_a = pipe_a
        self.pipe_b = pipe_b

        if str(pipe_a.device) != str(pipe_b.device):
            raise ValueError("Pipelines on different devices")
        if (
            pipe_a.scheduler.config.num_train_timesteps
            != pipe_b.scheduler.config.num_train_timesteps
        ):
            raise ValueError("Schedulers have different train timesteps")

    @staticmethod
    def _encode_prompt(
        pipe: DiffusionPipeline, prompt: str, batch_size: int, device: torch.device
    ):
        tokens = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        emb = pipe.text_encoder(tokens)[0]
        return emb.expand(batch_size, -1, -1)

    @staticmethod
    def project_vector(u, v, eps=1e-8):
        # compute in fp32 for stability
        u32 = u.float()
        v32 = v.float()

        u_flat = u32.view(u32.shape[0], -1)
        v_flat = v32.view(v32.shape[0], -1)

        dot_uv = (u_flat * v_flat).sum(dim=1, keepdim=True)
        dot_vv = (v_flat * v_flat).sum(dim=1, keepdim=True)

        # avoid tiny norms
        projection_factor = dot_uv / (dot_vv + eps)

        projection_factor = projection_factor.view(-1, 1, 1, 1)
        proj = projection_factor * v32
        return proj.to(dtype=v.dtype)

    @torch.no_grad()
    def __call__(
        self,
        prompt_a: str,
        prompt_b: str,
        lambda_: float,
        batch_size: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed_a: int = 32,
        seed_b: int | None = None,
        latents_a: torch.Tensor | None = None,
        latents_b: torch.Tensor | None = None,
    ):
        if seed_b is None:
            seed_b = seed_a

        pipe_a, pipe_b = self.pipe_a, self.pipe_b
        device = pipe_a.device
        dtype = pipe_a.unet.dtype

        # get encodings for propts, conditional and unconditional with prompt ""
        cond_a = self._encode_prompt(pipe_a, prompt_a, batch_size, device)
        cond_b = self._encode_prompt(pipe_b, prompt_b, batch_size, device)
        uncond_a = self._encode_prompt(pipe_a, "", batch_size, device)
        uncond_b = self._encode_prompt(pipe_b, "", batch_size, device)

        # TODO: test generators with different seeds, see how that changes performance
        gen_a = torch.Generator(device=device).manual_seed(seed_a)
        gen_b = torch.Generator(device=device).manual_seed(seed_b)

        # make some noise :D. initialize latents if not provided
        if latents_a is None:
            latents_a = CoupledDiffusion.make_initial_latents(
                pipe_a, batch_size, seed_a, device=device, dtype=dtype, generator=gen_a
            )

        if latents_b is None:
            latents_b = CoupledDiffusion.make_initial_latents(
                pipe_b, batch_size, seed_b, device=device, dtype=dtype, generator=gen_b
            )

        latents_a = latents_a.clone()
        latents_b = latents_b.clone()

        # set timesteps for schedulers to T
        pipe_a.scheduler.set_timesteps(num_inference_steps, device=device)
        pipe_b.scheduler.set_timesteps(num_inference_steps, device=device)

        # assuming both schedulers have same T since __init__() checks this
        timesteps = pipe_a.scheduler.timesteps

        def cfg_eps(pipe, latents_x, t_scalar, cond, uncond):
            eps_u = pipe.unet(latents_x, t_scalar, encoder_hidden_states=uncond).sample
            eps_c = pipe.unet(latents_x, t_scalar, encoder_hidden_states=cond).sample
            return eps_u + guidance_scale * (eps_c - eps_u)

        # reverse loop
        for i, t in enumerate(timesteps):
            eps_a = cfg_eps(pipe_a, latents_a, t, cond_a, uncond_a)
            eps_b = cfg_eps(pipe_b, latents_b, t, cond_b, uncond_b)

            # get x0 prediction and take DDPM step
            out_a = pipe_a.scheduler.step(eps_a, t, latents_a, generator=gen_a)
            out_b = pipe_b.scheduler.step(eps_b, t, latents_b, generator=gen_b)

            # calculate diff between the two x0s and project to get parallel and ortho
            delta_a = out_a.pred_original_sample - out_b.pred_original_sample
            delta_b = out_b.pred_original_sample - out_a.pred_original_sample

            parallel_a = CoupledDiffusion.project_vector(
                delta_a, out_a.pred_original_sample
            )
            parallel_b = CoupledDiffusion.project_vector(
                delta_b, out_b.pred_original_sample
            )

            ortho_a = delta_a - parallel_a
            ortho_b = delta_b - parallel_b

            latents_a = out_a.prev_sample
            latents_b = out_b.prev_sample

            # coupling based on algorithm 1d
            if i < len(timesteps) - 1:
                t_prev_int = int(timesteps[i + 1].item())
                alpha_prev = pipe_a.scheduler.alphas_cumprod[t_prev_int].to(
                    device=device, dtype=dtype
                )
                scale = (1.0 - alpha_prev).sqrt()
            else:
                scale = torch.tensor(0.0, device=device, dtype=dtype)

            eta = 0.1

            latents_a = latents_a - scale * lambda_ * (ortho_a + (eta) * parallel_a)
            latents_b = latents_b - scale * lambda_ * (ortho_b + (eta) * parallel_b)

        return latents_a, latents_b

    @staticmethod
    def make_initial_latents(
        pipe,
        batch_size,
        seed,
        device,
        dtype=torch.float32,
        generator: torch.Generator | None = None,
    ):
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(seed)

        return torch.randn(
            (
                batch_size,
                pipe.unet.config.in_channels,
                pipe.unet.config.sample_size,
                pipe.unet.config.sample_size,
            ),
            device=device,
            dtype=dtype,
            generator=generator,
        )


def retrieve_sd_pipeline(model: str = "manojb/stable-diffusion-2-1-base"):
    pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    return pipe


def decode_latents_to_pil(pipe: DiffusionPipeline, latents: torch.Tensor | None):
    with torch.no_grad():
        images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample

    images = (images / 2 + 0.5).clamp(0, 1)
    images = (images * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(img) for img in images]


def experimentA(args, pipe_a, pipe_b):
    sampler = CoupledDiffusion(pipe_a, pipe_b)

    device = pipe_a.device
    dtype = pipe_a.unet.dtype

    # create initial latents (different noise for a and b)
    init_a = CoupledDiffusion.make_initial_latents(pipe_a, 1, 440, device, dtype)
    init_b = CoupledDiffusion.make_initial_latents(pipe_b, 1, 440, device, dtype)

    # ensure output dir exists (fixes FileNotFoundError in Experiment B)
    os.makedirs(args.output_dir, exist_ok=True)

    # experiment ste 1p : coupled run (lambda > 0)
    print(f"Running coupled diffusion (lambda={args.lambda_})...")
    coupled_a, coupled_b = sampler(
        args.prompt_a,
        args.prompt_b,
        lambda_=args.lambda_,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed_a=args.seed_a,
        seed_b=args.seed_b,
        latents_a=init_a,
        latents_b=init_b,
    )

    # experiment step 2: uncoupled run (lambda = 0) with same initial latents
    print("Running uncoupled diffusion (lambda=0)...")
    uncoupled_a, uncoupled_b = sampler(
        args.prompt_a,
        args.prompt_b,
        lambda_=0.0,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed_a=args.seed_a,
        seed_b=args.seed_b,
        latents_a=init_a,
        latents_b=init_b,
    )

    # decode and save
    decode_latents_to_pil(pipe_a, coupled_a)[0].save(f"{args.output_dir}/coupled_a.png")
    decode_latents_to_pil(pipe_b, coupled_b)[0].save(f"{args.output_dir}/coupled_b.png")
    decode_latents_to_pil(pipe_a, uncoupled_a)[0].save(
        f"{args.output_dir}/uncoupled_a.png"
    )
    decode_latents_to_pil(pipe_b, uncoupled_b)[0].save(
        f"{args.output_dir}/uncoupled_b.png"
    )

    print("Saved outputs")


def main():
    parser = ArgumentParser()
    parser.add_argument("--prompt_a", type=str, default=None)
    parser.add_argument("--prompt_b", type=str, default=None)
    parser.add_argument("--lambda_", type=float, default=0.016)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed_a", type=int, default=32)
    parser.add_argument("--seed_b", type=int, default=64)
    parser.add_argument("--model", type=str, default="manojb/stable-diffusion-2-1-base")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/fs/nexus-scratch/ltahboub/learning-diffusion/CoupledDiffusion/inference/experiment_outputs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pipe_a = retrieve_sd_pipeline(args.model)
    pipe_b = retrieve_sd_pipeline(args.model)

    if args.experiment == "A":
        if args.prompt_a is None or args.prompt_b is None:
            raise ValueError("Prompts can't be None for Experiment A")
        experimentA(args, pipe_a, pipe_b)
    # elif args.experiment == "B":
    #     experimentB(args, pipe_a, pipe_b)
    else:
        raise ValueError("Experiment type not found.")


if __name__ == "__main__":
    main()
