"""
DRSformer De-raining Utilities

This module provides functions to load and use the DRSformer model for image de-raining.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from runpy import run_path

# DRSformer source path
DRSFORMER_SRC_PATH = r'E:\Python\DLCV\DRSformer'

# Add DRSformer to path
if DRSFORMER_SRC_PATH not in sys.path:
    sys.path.insert(0, DRSFORMER_SRC_PATH)


def load_drsformer_model(model_path, device='cuda'):
    """
    Load DRSformer model from checkpoint.
    
    Args:
        model_path (str): Path to the model checkpoint (.pth file)
        device (str): Device to load model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded DRSformer model in eval mode
    """
    # Model parameters (from DRSformer default config)
    parameters = {
        'inp_channels': 3,
        'out_channels': 3,
        'dim': 48,
        'num_blocks': [4, 6, 6, 8],
        'heads': [1, 2, 4, 8],
        'ffn_expansion_factor': 2.66,
        'bias': False,
        'LayerNorm_type': 'WithBias'
    }
    
    # Load architecture
    arch_path = os.path.join(DRSFORMER_SRC_PATH, 'basicsr', 'models', 'archs', 'DRSformer_arch.py')
    load_arch = run_path(arch_path)
    model = load_arch['DRSformer'](**parameters)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['params'])
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"DRSformer model loaded from: {model_path}")
    return model


def derain_image(image, model, device='cuda', img_multiple_of=8, tile=None, tile_overlap=32):
    """
    Apply de-raining to a PIL Image using DRSformer.
    
    Args:
        image (PIL.Image): Input rainy image
        model: DRSformer model
        device (str): Device to run inference on
        img_multiple_of (int): Padding multiple (default: 8)
        tile (int): Tile size for tiled inference (e.g., 512, 720). 
                   None means testing on the original resolution (slower for large images).
                   Using tiles (e.g., 512 or 720) significantly speeds up inference.
        tile_overlap (int): Overlapping of different tiles (default: 32)
    
    Returns:
        PIL.Image: De-rained image
    """
    # Convert PIL Image to numpy array
    img_np = np.array(image)
    
    # Convert to torch tensor [C, H, W] and normalize to [0, 1]
    input_tensor = torch.from_numpy(img_np).float().div(255.).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Pad the input if not multiple of img_multiple_of
    height, width = input_tensor.shape[2], input_tensor.shape[3]
    H = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of
    W = ((width + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - height if height % img_multiple_of != 0 else 0
    padw = W - width if width % img_multiple_of != 0 else 0
    input_tensor = F.pad(input_tensor, (0, padw, 0, padh), 'reflect')
    
    # Run inference
    with torch.no_grad():
        if tile is None:
            # Testing on the original resolution image (slower but more accurate)
            restored = model(input_tensor)
        else:
            # Test the image tile by tile (faster for large images)
            b, c, h, w = input_tensor.shape
            tile_size = min(tile, h, w)
            assert tile_size % 8 == 0, "tile size should be multiple of 8"
            
            stride = tile_size - tile_overlap
            h_idx_list = list(range(0, h - tile_size, stride)) + [h - tile_size]
            w_idx_list = list(range(0, w - tile_size, stride)) + [w - tile_size]
            E = torch.zeros(b, c, h, w).type_as(input_tensor)
            W = torch.zeros_like(E)
            
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_tensor[..., h_idx:h_idx + tile_size, w_idx:w_idx + tile_size]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)
                    
                    E[..., h_idx:(h_idx + tile_size), w_idx:(w_idx + tile_size)].add_(out_patch)
                    W[..., h_idx:(h_idx + tile_size), w_idx:(w_idx + tile_size)].add_(out_patch_mask)
            
            restored = E.div_(W)
        
        restored = torch.clamp(restored, 0, 1)
    
    # Remove padding
    restored = restored[:, :, :height, :width]
    
    # Convert back to numpy and PIL Image
    restored_np = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
    restored_np = np.clip(restored_np * 255.0, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image
    derained_pil = Image.fromarray(restored_np, mode='RGB')
    
    return derained_pil


def compare_derain_methods(image, spdnet_model, drsformer_model, device='cuda', tile=512):
    """
    Compare SPDNet and DRSformer de-raining on the same image.
    
    Args:
        image (PIL.Image): Input rainy image
        spdnet_model: SPDNet model
        drsformer_model: DRSformer model
        device (str): Device to run inference on
        tile (int): Tile size for DRSformer inference (default: 512)
    
    Returns:
        dict: Dictionary with 'spdnet' and 'drsformer' de-rained images
    """
    from utils.spdnet_utils import derain_image as spdnet_derain
    
    results = {}
    
    # Apply SPDNet
    print("Applying SPDNet de-raining...")
    results['spdnet'] = spdnet_derain(image, spdnet_model, device)
    
    # Apply DRSformer
    print("Applying DRSformer de-raining...")
    results['drsformer'] = derain_image(image, drsformer_model, device, tile=tile)
    
    return results
