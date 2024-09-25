# Copyright (c) 2024, Huangjie Zheng. All rights reserved.
#
# This work is licensed under a MIT License.

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from diffusers.models import AutoencoderKL

# utility functions
def find_latest_checkpoint(directory):
    """
    Finds the latest network checkpoint file in a directory and its subdirectories.
    
    :param directory: The path to the directory to search in.
    :return: The path to the latest checkpoint file, or None if no such file is found.
    """
    latest_file = None
    latest_number = -1
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("network-snapshot-") and file.endswith(".pkl"):
                # Extract the number from the file name
                number_part = file[len("network-snapshot-"):-len(".pkl")]
                try:
                    number = int(number_part)
                    if number > latest_number:
                        latest_number = number
                        latest_file = os.path.join(root, file)
                except ValueError:
                    # If the number part is not an integer, ignore this file
                    continue
    return latest_file

def compress_to_npz(folder_path, npz_path):
    file_names = os.listdir(folder_path)
    file_names = [file_name for file_name in file_names if file_name.endswith(('.png', '.jpg', '.jpeg'))]

    samples = []

    for file_name in tqdm(file_names, desc="Compressing images"):
        file_path = os.path.join(folder_path, file_name)
        
        image = Image.open(file_path)
        image_array = np.asarray(image).astype(np.uint8)
        
        samples.append(image_array)
    samples = np.stack(samples)
    np.savez(npz_path, arr_0=samples)
    dist.print0(f"Images from folder {folder_path} have been saved as {npz_path}")



#----------------------------------------------------------------------------
# EDM sampler (https://arxiv.org/abs/2206.00364) with option to activate which bricks are used in sampling.
def edm_lego_sampler(
    net, latents, class_labels=None, cfg_scale=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7, skip_ratio=0.6,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, use_skip=False, use_full_channels=True,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    return_stage_idx = -1 # initialize return_stage_idx
    
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        if use_skip and i > int(num_steps * skip_ratio):
            return_stage_idx = net.model.num_bricks - 1 # skip the last brick for fast sampling

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat * torch.ones(x_hat.shape[0],).to(x_hat.device), class_labels, cfg_scale, use_full_channels=use_full_channels, return_stage_idx=return_stage_idx).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next * torch.ones(x_hat.shape[0],).to(x_hat.device), class_labels, cfg_scale, use_full_channels=use_full_channels, return_stage_idx=return_stage_idx).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--img_resolution', 'img_resolution',      help='image resolution', metavar='INT',                    type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--img_channels', 'img_channels',      help='image channels', metavar='INT',                          type=click.IntRange(min=1), default=3, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)
@click.option('--cfg_scale', 'cfg_scale',  help='Cfg scale parameter', metavar='FLOAT',                             type=float, default=None, show_default=True)
@click.option('--skip_bricks', 'skip_bricks',  help='Returning stage index', metavar='INT',                         type=bool, default=False, show_default=True)
@click.option("--fc", 'use_full_channels', help="use full channels in cfg sampling", metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--skip_ratio', 'skip_ratio',      help='Start from how many timesteps we skip LEGO bricks', metavar='INT',                          type=click.FloatRange(min=0.0, max=1.0), default=0.6, show_default=True)
@click.option('--height_ratio', 'height_ratio',      help='Ratio to sample smaller/larger content', metavar='INT',                          type=click.FloatRange(min=1.0), default=1.0, show_default=True)
@click.option('--width_ratio', 'width_ratio',      help='Ratio to sample smaller/larger content', metavar='INT',                          type=click.FloatRange(min=1.0), default=1.0, show_default=True)

@click.option('--vae', 'use_vae',         help='whether use SD VAE in generation', metavar='STR',                   type=bool, default=False, show_default=True)
@click.option('--npz', 'compress_npz',         help='whether use SD VAE in generation', metavar='STR',                   type=bool, default=False, show_default=True)

def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, cfg_scale, use_full_channels, skip_bricks, img_resolution, img_channels, height_ratio, width_ratio, skip_ratio, use_vae, compress_npz, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=LEGO-L-PG-64.pkl

    \b
    # Generate 1024 images using 2 GPUs with SD VAE and compress output folder to out.npz
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-1023 --batch=64 \\
        --network=LEGO-XL-U-256.pkl --npz=True --vae=True
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # load vae
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device) if use_vae else None
    
    # Load network.
    if os.path.isdir(network_pkl):
        dist.print0(f'Looking for the latest network from "{network_pkl}"...')
        network_pkl = find_latest_checkpoint(network_pkl)
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema']
    net.eval().to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, int(net.img_resolution*height_ratio), int(net.img_resolution*width_ratio)], device=device)
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        sampler_fn =  edm_lego_sampler
        with torch.no_grad():
            images = sampler_fn(net, latents, class_labels, cfg_scale, randn_like=rnd.randn_like, use_full_channels=use_full_channels, use_skip=skip_bricks, skip_ratio=skip_ratio, **sampler_kwargs)
            images = vae.decode((images / 0.18215).float()).sample if use_vae else images

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # Done.
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        if compress_npz:
            compress_to_npz(outdir, outdir + ".npz")
    torch.distributed.barrier()
    dist.print0('Done.')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
