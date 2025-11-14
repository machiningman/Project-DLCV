"""Model utilities for loading and configuring RT-DETR models"""

from transformers import RTDetrImageProcessor, RTDetrConfig, RTDetrForObjectDetection


def load_model_and_processor(model_name="PekingU/rtdetr_r18vd", num_labels=80):
    """
    Load RT-DETR model and processor.
    
    Args:
        model_name: HuggingFace model name or path (default: "PekingU/rtdetr_r18vd")
        num_labels: Number of object detection classes (default: 80 for COCO)
        
    Returns:
        Tuple of (model, processor)
    """
    # Load processor
    processor = RTDetrImageProcessor.from_pretrained(model_name)
    
    # Load model configuration
    config = RTDetrConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    
    # Load model
    model = RTDetrForObjectDetection.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=False
    )
    
    return model, processor


def print_model_info(model):
    """Print model parameter information"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{trainable}/{total} trainable params ({trainable/total:.2%})")
