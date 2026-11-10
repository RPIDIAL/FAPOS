"""Model loading and landmark prediction (VQ-VAE latent + conditional diffusion transformer)."""
import numpy as np
import torch

from diffusion import create_diffusion
from model import ConDiT_models, VQVAE_Transformer_models


def _strip(state):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def load_models(vqvae_path, fapos_path, device="cpu", vae_model="VQVAET-T", dit_model="ConDiT-T",
                num_cbembed=512, dim_cbembed=4, cond_depth=6, point_size=282):
    """Build the networks with the published FAPOS configuration and load the checkpoints."""
    vqvae = VQVAE_Transformer_models[vae_model](input_size=point_size, in_channels=3, num_cbembed=num_cbembed,
                                                dim_cbembed=dim_cbembed).to(device)
    vqvae.load_state_dict(_strip(torch.load(vqvae_path, map_location="cpu", weights_only=False)["model"]))
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    dit = ConDiT_models[dit_model](input_size=point_size, in_channels=dim_cbembed, cond_depth=cond_depth,
                                   learn_sigma=False).to(device)
    dit.load_state_dict(_strip(torch.load(fapos_path, map_location="cpu", weights_only=False)["model"]))
    dit.eval()
    return vqvae, dit


@torch.no_grad()
def predict_normalized(vqvae, dit, cond_norm282, device="cpu", seed=0, steps=1000):
    """cond_norm282: (282, 3) or (N, 282, 3) normalized pre-operative landmarks -> normalized prediction, same shape."""
    arr = np.asarray(cond_norm282)
    single = arr.ndim == 2
    cond = torch.tensor(arr[None] if single else arr, dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    diffusion = create_diffusion(str(steps), learn_sigma=False)

    h = vqvae.p_embedder(cond) + vqvae.pos_embed
    for block in vqvae.encoder_blocks:
        h = block(h)
    h = vqvae.encod_emb(h)

    noise = torch.randn(h.shape[0], h.shape[1], h.shape[2], device=device)
    samples = diffusion.p_sample_loop(dit.forward, noise.shape, cond=h, noise=noise, clip_denoised=False,
                                      model_kwargs=None, progress=False, device=device)
    quantized, _ = vqvae.vq_layer(samples.transpose(1, 2).contiguous())
    z = vqvae.decod_emb(quantized) + vqvae.pos_embed
    for block in vqvae.decoder_blocks:
        z = block(z)
    recon = vqvae.final_layer(z)[:, :, :3].cpu().numpy().astype(float)
    return recon[0] if single else recon
