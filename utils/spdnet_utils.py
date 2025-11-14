"""SPDNet utilities for image de-raining preprocessing

Note: SPDNet model has hardcoded CUDA calls in its architecture.
If you need to run on CPU only, you would need to modify the SPDNet source code
to remove the .cuda() calls in SPDNet/src/model/spdnet.py (lines 147-148).
"""

import os
import sys
import torch
import numpy as np
from PIL import Image


# Add SPDNet source directory to path
# SPDNet is located at E:\Python\DLCV\SPDNet, not in the Project folder
SPDNET_SRC_PATH = r'E:\Python\DLCV\SPDNet\src'
if SPDNET_SRC_PATH not in sys.path:
    sys.path.insert(0, SPDNET_SRC_PATH)


def load_spdnet_model(model_path, device='cuda', n_feats=32, n_resblocks=3):
    """
    Load SPDNet model for de-raining.
    
    Args:
        model_path: Path to the pre-trained SPDNet model (.pt file)
        device: Device to load the model on ('cuda' or 'cpu')
               Note: SPDNet requires CUDA. CPU-only mode requires modifying SPDNet source.
        n_feats: Number of feature maps (default: 32)
        n_resblocks: Number of residual blocks (default: 3)
        
    Returns:
        SPDNet model ready for inference
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If CUDA is not available but device='cuda'
    """
    # Import SPDNet model
    from model import spdnet
    
    # Check CUDA availability
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but CUDA is not available. "
            "SPDNet requires CUDA for inference due to hardcoded .cuda() calls. "
            "To use CPU, you must modify SPDNet/src/model/spdnet.py."
        )
    
    # Create a mock args object with required attributes
    class Args:
        def __init__(self):
            self.n_feats = n_feats
            self.n_resblocks = n_resblocks
    
    args = Args()
    
    # Load the model (this will use CUDA due to hardcoded .cuda() in SPDNet)
    model = spdnet.make_model(args)
    
    # Load pre-trained weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"SPDNet model loaded from: {model_path}")
    else:
        raise FileNotFoundError(f"SPDNet model not found at: {model_path}")
    
    # Move model to the correct device
    model = model.to(device)
    model.eval()
    
    return model


def derain_image(image, model, device='cuda', rgb_range=255):
    """
    De-rain a single image using SPDNet.
    
    Args:
        image: PIL Image object or numpy array (H, W, C) in RGB format
        model: SPDNet model
        device: Device to run inference on
        rgb_range: RGB value range (default: 255)
        
    Returns:
        De-rained PIL Image object
    """
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image).astype(np.float32)
    else:
        image_np = image.astype(np.float32)
    
    # Ensure RGB format (H, W, C)
    if len(image_np.shape) == 2:
        # Grayscale to RGB
        image_np = np.stack([image_np] * 3, axis=-1)
    
    # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    # SPDNet expects input in range [0, rgb_range], NOT normalized to [0, 1]
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()
    image_tensor = image_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        # SPDNet returns (out3, out2, out1), we use out3 (final output)
        derained_tensor, _, _ = model(image_tensor)
    
    # Convert back to numpy: (1, C, H, W) -> (C, H, W) -> (H, W, C)
    derained_np = derained_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Clip to valid range (SPDNet output is in range [0, rgb_range])
    derained_np = np.clip(derained_np, 0, rgb_range)
    derained_np = derained_np.astype(np.uint8)
    
    # Convert back to PIL Image
    derained_image = Image.fromarray(derained_np)
    
    return derained_image


def quantize(img, rgb_range=255):
    """
    Quantize image tensor to valid range.
    
    Args:
        img: Image tensor
        rgb_range: RGB value range
        
    Returns:
        Quantized image tensor
    """
    pixel_range = rgb_range
    return img.mul(pixel_range).clamp(0, rgb_range).round().div(pixel_range)
