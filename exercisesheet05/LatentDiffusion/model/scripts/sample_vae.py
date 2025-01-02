import os
import torch as th
import numpy as np
from PIL import Image

import pytorch_lightning as pl

from data.lightning_ffhq import FFHQDataModule
from model.autoencoder.ffhq import FfhqAutoencoder
from model.lightning.vae_trainer import VAETrainerModule
from utils.configuration import Configuration
import torch

def cosine_beta_schedule(num_timesteps, s=0.008):
    # Timesteps are from 0 to T-1
    alphas_cumprod = torch.linspace(0, num_timesteps, num_timesteps+1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((alphas_cumprod / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0, 0.999)

def sample_vae(cfg: Configuration, checkpoint_path, output_path='samples', num_samples=1, device=0, noise=True):
    os.makedirs(output_path, exist_ok=True)

    # Set device
    device = th.device("cpu") #f"cuda:{device}" if th.cuda.is_available() else "cpu")

    # Initialize the model
    if cfg.model.data == 'ffhq':
        model_net = FfhqAutoencoder(cfg=cfg.model)
    else:
        raise NotImplementedError(f"Data {cfg.model.data} not implemented")

    # Load the model checkpoint
    model = VAETrainerModule.load_from_checkpoint(
        checkpoint_path,
        cfg=cfg,
        model=model_net,
        strict=False,
        map_location=device
    ).to(device)
    model.eval()

    # Initialize data module and get the test dataset
    data_module = FFHQDataModule(cfg)
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()

    # Sample pairs
    with th.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx >= num_samples:
                break

            # Get input and send to device
            input_image = batch[0].to(device)
            if noise:
                n_steps = 15
                schedules = {"cosine": cosine_beta_schedule(n_steps), "linear": torch.linspace(0, 1, n_steps)}
                for noise_type, betas in schedules.items():
                    z, _ = model.net.encode(input_image[0])
                    alpha_cumprod = torch.cumprod(1 - betas, dim=0)
                    images = []
                    for t in range(n_steps):
                        z_t = torch.sqrt(alpha_cumprod[t])*z + torch.randn_like(z)*torch.sqrt(1-alpha_cumprod[t])
                        decoded = model.net.output_postprocessing(model.net.decode(z_t))
                        image = (decoded.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        images.append(image)
                    Image.fromarray(np.concatenate(images, axis=1)).save(os.path.join(output_path, f'sample_noised_{noise_type}.png'))
            else:
                # Get reconstructed image
                results = model(input_image)
            output_image = results['output']
            input_image  = results['input']

            # Process and save input-output pairs
            for i in range(input_image.size(0)):
                inp_img = input_image[i].cpu()
                out_img = output_image[i].cpu()

                # Convert to numpy and scale to [0, 255]
                np_input_img = inp_img.numpy().transpose(1, 2, 0) * 255
                np_output_img = out_img.numpy().transpose(1, 2, 0) * 255

                np_input_img = np_input_img.astype(np.uint8)
                np_output_img = np_output_img.astype(np.uint8)

                # Concatenate input and output images side-by-side
                combined_img = np.concatenate((np_input_img, np_output_img), axis=1)

                # Convert to PIL and save
                pil_image = Image.fromarray(combined_img)
                pil_image.save(os.path.join(output_path, f'sample_{idx}_{i}.png'))

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("-load", "--load", required=True, type=str, help="Path to the model checkpoint")
    parser.add_argument("-output", "--output", default="vae-samples", type=str, help="Output directory for samples")
    parser.add_argument("-num_samples", "--num_samples", default=1, type=int, help="Number of samples to generate")
    parser.add_argument("-device", "--device", default=0, type=int, help="CUDA device ID")
    parser.add_argument("-seed", "--seed", default=1234, type=int, help="Random seed")

    args = parser.parse_args(sys.argv[1:])
    cfg = Configuration(args.cfg)

    cfg.seed = args.seed
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)

    sample_vae(
        cfg=cfg,
        checkpoint_path=args.load,
        output_path=args.output,
        num_samples=args.num_samples,
        device=args.device
    )
