from typing import List, Optional
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from einops import rearrange

def _to_pil(image: torch.Tensor | Image.Image, target_wh: tuple[int, int], mean: tuple[float, float, float], std: tuple[float, float, float]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.resize(target_wh)
    assert image.ndim == 3  # [channel, height, width]
    image_denormed = (image.detach().cpu() * torch.tensor(std)[:, None, None]) \
                    + torch.tensor(mean)[:, None, None]
    arr = (image_denormed.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr).resize(target_wh)

def visualize(
    images: torch.Tensor | Image.Image,
    heatmaps: torch.Tensor,
    mean_std: tuple[tuple[float, float, float], tuple[float, float, float]],
    alpha: float = 0.7,
    text_prompts: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
    title: Optional[str] = None,
):
    """
    Overlay heatmaps on the input image.
    Args:
        images: PIL or normalized tensor [B,C,H,W]
        heatmaps: [N, H, W] or [1,N,H,W] tensor in [0,1]
        alpha: overlay strength
        text_prompts: optional titles per heatmap
        save_dir: optional directory to save pngs
        title: optional title for the original image
    """
    assert images.ndim == 4  # [batch_size, channel, height, width]
    assert heatmaps.ndim == 4   # [batch_size, num_concepts, H, W]
    H, W = heatmaps.shape[-2:]
    num_images = heatmaps.shape[0]
    num_concepts = heatmaps.shape[1]

    if text_prompts is None:
        text_prompts = [str(i) for i in range(num_concepts)]
    pil_imgs = [_to_pil(img, (H, W), mean=mean_std[0], std=mean_std[1]) for img in images]
    img_cvs = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in pil_imgs]
    heatmaps_np = (heatmaps.detach().cpu().numpy() * 255).astype("uint8")   # [N, num_concepts, H, W]
    
    fig, axes = plt.subplots(num_images,
                             1 + heatmaps_np.shape[1],
                             figsize=(4 * (1 + num_concepts), 4),
                             squeeze=False)
    axes = np.atleast_1d(axes)
    for i, (pil_img, img_cv) in enumerate(zip(pil_imgs, img_cvs)):
        axes[i, 0].imshow(pil_img)
        axes[i, 0].axis("off")
        # if title:
        #     axes[i, 0].set_title(title)

        for j, hm in enumerate(heatmaps_np[i]):
            hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            overlay = (1 - alpha) * img_cv + alpha * hm
            ov_rgb = cv2.cvtColor(overlay.astype("uint8"), cv2.COLOR_BGR2RGB)
            axes[i, j + 1].imshow(ov_rgb)
            axes[i, j + 1].axis("off")
            if j < len(text_prompts):
                axes[i, j + 1].set_title(str(text_prompts[j]))
    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"heatmap_{title}.png"
        # Image.fromarray(ov_rgb.astype("uint8")).save(out_path)
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()
    return fig

def visualize_layerwise_maps(
        images: torch.Tensor,
        heatmaps: List[torch.Tensor],
        mean_std: tuple[tuple[float, float, float], tuple[float, float, float]],
        alpha: float = 0.7,
        text_prompts: Optional[List[str]] = None,
        save_dir: Optional[Path] = None,
        title: Optional[str] = None
    ):
    """Visualize the stored maps across all requested layers.
    Args:
        images: Normalized tensor [B,C,H,W]
        heatmaps: [H, W, B, M] * num_layers
        alpha: overlay strength
        text_prompts: optional titles per heatmap
        save_dir: optional directory to save pngs
        title: optional title for the original image
    """
    assert images.ndim == 4
    H, W = images.shape[-2], images.shape[-1]
    scale = H // heatmaps[0].shape[0]
    heatmaps = [rearrange(hm, 'h w b m -> b m h w') for hm in heatmaps]
    heatmaps = [F.interpolate(hm, scale_factor=scale, mode='bilinear') for hm in heatmaps]
    heatmaps = torch.stack(heatmaps, dim=0)  # [num_layers, B, M, H, W]

    assert heatmaps.ndim == 5
    assert heatmaps.shape[1] == images.shape[0]  # batch size
    
    num_images = heatmaps.shape[1]
    num_concepts = heatmaps.shape[2]
    num_layers = heatmaps.shape[0]

    heatmaps = (heatmaps - heatmaps.min()) / (heatmaps.max() - heatmaps.min() + 1e-8)
    heatmaps_np = (heatmaps.detach().cpu().numpy() * 255).astype("uint8")  

    fig, axes = plt.subplots(
        num_images*num_concepts,
        num_layers+1,
        figsize=(4*(1 + num_layers), 4*num_images*num_concepts),
        squeeze=False)
    axes = np.atleast_1d(axes)
    for i in range(num_images):
        pil_img = _to_pil(images[i], (H, W), mean=mean_std[0], std=mean_std[1])
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        for j in range(num_concepts):
            axes[i*num_concepts + j, 0].imshow(pil_img)
            axes[i*num_concepts + j, 0].axis("off")
            if text_prompts is not None and j < len(text_prompts):
                axes[i*num_concepts + j, 0].set_title(str(text_prompts[j]))
            for k in range(num_layers):
                hm = heatmaps_np[k, i, j, :, :][:, :, None] # [H, W, 1]
                hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                overlay = (1 - alpha) * img_cv + alpha * hm_color
                ov_rgb = cv2.cvtColor(overlay.astype("uint8"), cv2.COLOR_BGR2RGB)
                axes[i*num_concepts + j, k + 1].imshow(ov_rgb)
                axes[i*num_concepts + j, k + 1].axis("off")
                axes[i*num_concepts + j, k + 1].set_title(f"Layer {k}")
    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"heatmap_{title}.png"
        # Image.fromarray(ov_rgb.astype("uint8")).save(out_path)
        plt.savefig(out_path)
        plt.close()
    else:
        plt.show()
