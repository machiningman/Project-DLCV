"""
Rain Detector Module for Conditional De-raining

This module provides a lightweight binary classifier to detect rain in images.
Used to conditionally apply de-raining only when needed, improving inference speed.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class RainDetector(nn.Module):
    """
    Lightweight rain detector using MobileNetV3.
    
    Binary classifier that predicts rain probability [0, 1] for input images.
    Designed for fast inference (~3-5ms on GPU).
    
    Architecture:
        - Backbone: MobileNetV3-Small (pretrained on ImageNet)
        - Head: 2-layer MLP with dropout
        - Output: Rain probability (sigmoid)
    """
    
    def __init__(self, pretrained=True, dropout=0.2):
        """
        Args:
            pretrained: Use ImageNet pretrained weights for backbone
            dropout: Dropout rate in classifier head
        """
        super().__init__()
        
        # Load pretrained MobileNetV3-Small (efficient backbone)
        if pretrained:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            base_model = mobilenet_v3_small(weights=weights)
        else:
            base_model = mobilenet_v3_small(weights=None)
        
        # Feature extractor (all layers except final classifier)
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Binary classification head
        # MobileNetV3-Small outputs 576 channels
        # NOTE: No sigmoid here - use BCEWithLogitsLoss for training
        self.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.Hardswish(inplace=True),  # MobileNetV3 uses Hardswish
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(128, 1)
            # No Sigmoid! BCEWithLogitsLoss is safer with mixed precision
        )
        
        # Initialize classifier head
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, H, W), normalized to [0, 1]
        
        Returns:
            logits: Raw logits for each image (B,) - use with BCEWithLogitsLoss
        """
        # Extract features
        x = self.features(x)  # (B, 576, H', W')
        x = self.avgpool(x)   # (B, 576, 1, 1)
        x = torch.flatten(x, 1)  # (B, 576)
        
        # Binary classification (raw logits, no sigmoid)
        logits = self.classifier(x)  # (B, 1)
        logits = logits.squeeze(-1)  # (B,)
        
        return logits
    
    def predict(self, x, threshold=0.5):
        """
        Predict binary rain labels.
        
        Args:
            x: Input images (B, 3, H, W)
            threshold: Classification threshold
        
        Returns:
            is_rainy: Binary predictions (B,) - True if rainy
            rain_scores: Probabilities (B,) after sigmoid
        """
        logits = self.forward(x)
        rain_scores = torch.sigmoid(logits)  # Convert logits to probabilities
        is_rainy = rain_scores > threshold
        return is_rainy, rain_scores


def load_rain_detector(checkpoint_path, device='cuda'):
    """
    Load pretrained rain detector from checkpoint.
    
    Args:
        checkpoint_path: Path to saved model (.pt or .pth)
        device: Device to load model on
    
    Returns:
        Rain detector model ready for inference
    """
    model = RainDetector(pretrained=False)  # Don't load ImageNet weights
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


class RainDetectionDataset(torch.utils.data.Dataset):
    """
    Dataset for training rain detector (binary classification).
    
    Converts object detection dataset to binary classification format.
    """
    
    def __init__(self, coco_dataset, transform=None):
        """
        Args:
            coco_dataset: COCO-style dataset with 'image' and 'domain' fields
            transform: Optional transforms (should include normalization)
        """
        self.dataset = coco_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor (3, H, W)
            label: Binary label (0=clean, 1=rainy)
        """
        sample = self.dataset[idx]

        # Support multiple dataset formats:
        # - supervision.DetectionDataset returns (image_id, image_np, annotations) - 3-tuple!
        # - dict-like samples with keys 'image' and optional 'domain'
        if isinstance(sample, dict):
            image = sample.get('image') or sample.get('img') or sample.get('image_array')
            domain = sample.get('domain', None)
        elif isinstance(sample, (tuple, list)) and len(sample) == 3:
            # Supervision format: (image_id, image, annotations)
            image_id, image, annotations = sample
            domain = None
        elif isinstance(sample, (tuple, list)):
            # Fallback: assume (image, annotations) or just (image,)
            image = sample[0]
            annotations = sample[1] if len(sample) > 1 else None
            domain = None
        else:
            # Fallback: sample itself is image
            image = sample
            domain = None

        # If domain is not provided in annotation/dict, try to infer from dataset path
        if domain is None:
            dataset_paths = getattr(self.dataset, 'image_paths', None) or getattr(self.dataset, 'images', None)
            if dataset_paths is not None and idx < len(dataset_paths):
                p = str(dataset_paths[idx]).lower()
                # Fix: 'train' contains 'rain', so we must be more specific
                # 'coco_rain' is the folder name for rainy images
                domain = 'rainy' if 'coco_rain' in p else 'clean'
            else:
                domain = 'clean'

        label = 1.0 if 'rain' in domain.lower() else 0.0
        
        # Apply transforms if provided
        if self.transform:
            from PIL import Image
            import os
            import cv2
            import numpy as np
            
            # Ensure image is PIL for torchvision transforms
            if isinstance(image, (str, os.PathLike)):
                image = Image.open(str(image)).convert('RGB')
            elif not isinstance(image, Image.Image):
                # Convert numpy array to PIL Image
                if isinstance(image, np.ndarray):
                    # Assume BGR from OpenCV/Supervision if 3 channels
                    if image.ndim == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                else:
                    arr = np.array(image)
                    if arr.ndim == 3 and arr.shape[2] == 3:
                        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(arr)
            
            image = self.transform(image)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.float32)
        }


def get_rain_detection_transforms():
    """
    """
    from torchvision import transforms
    
    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # MobileNet input size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform
