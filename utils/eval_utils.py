"""Evaluation utilities for model inference and visualization"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from matplotlib.patches import Rectangle


# COCO class names
COCO_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def run_inference(image_path, model, processor, device, threshold=0.3, 
                  derain_model=None, derain_device=None):
    """
    Run inference on a single image with optional de-raining preprocessing.
    
    Args:
        image_path: Path to the image file
        model: RT-DETR model
        processor: RTDetrImageProcessor
        device: Device to run inference on
        threshold: Confidence threshold for detections
        derain_model: Optional SPDNet model for de-raining (default: None)
        derain_device: Device for de-raining model (default: None, uses device)
        
    Returns:
        Tuple of (results, image_size)
    """
    image = Image.open(image_path).convert("RGB")
    
    # Optional de-raining preprocessing
    if derain_model is not None:
        try:
            from .spdnet_utils import derain_image
        except ImportError:
            from spdnet_utils import derain_image
        if derain_device is None:
            derain_device = device
        image = derain_image(image, derain_model, derain_device)
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs, 
        target_sizes=target_sizes, 
        threshold=threshold
    )[0]
    
    return results, image.size


def build_label_mapping(class_names, coco_gt):
    """
    Build mapping from model class indices to COCO category IDs.
    
    Args:
        class_names: List of class names from the model
        coco_gt: COCO ground truth object
        
    Returns:
        List mapping label indices to COCO category IDs
    """
    coco_cats = coco_gt.loadCats(coco_gt.getCatIds())
    coco_name_to_id = {c['name']: c['id'] for c in coco_cats}
    
    label_to_coco_id = []
    missing_names = []
    
    for i, name in enumerate(class_names):
        coco_id = coco_name_to_id.get(name)
        label_to_coco_id.append(coco_id)
        if coco_id is None:
            missing_names.append((i, name))
    
    if missing_names:
        print(f"Warning: {len(missing_names)} class names not found in COCO categories")
    
    return label_to_coco_id


def generate_predictions(model, processor, device, coco_gt, image_ids, val_images_dir, 
                         label_to_coco_id, class_names, threshold=0.01,
                         derain_model=None, derain_device=None, use_drsformer=False,
                         drsformer_tile=512, drsformer_tile_overlap=32):
    """
    Generate predictions for all validation images in COCO format.
    
    Args:
        model: RT-DETR model
        processor: RTDetrImageProcessor
        device: Device to run inference on
        coco_gt: COCO ground truth object
        image_ids: List of COCO image IDs
        val_images_dir: Directory containing validation images
        label_to_coco_id: Mapping from label indices to COCO category IDs
        class_names: List of class names
        threshold: Confidence threshold for detections
        derain_model: Optional de-raining model (SPDNet or DRSformer)
        derain_device: Device for de-raining model (default: None, uses device)
        use_drsformer: Set to True if derain_model is DRSformer (default: False for SPDNet)
        drsformer_tile: Tile size for DRSformer inference (default: 512, use None for full image)
        drsformer_tile_overlap: Overlap between tiles for DRSformer (default: 32)
        
    Returns:
        List of predictions in COCO format
    """
    predictions_mapped = []
    skipped_count = 0
    total_preds = 0
    
    # Determine derain method name for display
    if derain_model is not None:
        derain_method = "DRSformer" if use_drsformer else "SPDNet"
        tile_info = f" (tile={drsformer_tile})" if use_drsformer and drsformer_tile else ""
        derain_msg = f" with {derain_method}{tile_info} de-raining preprocessing"
    else:
        derain_msg = ""
    
    print(f"\nRunning inference{derain_msg} and mapping labels to COCO ids...")
    for img_id in tqdm(image_ids, desc="Images"):
        # Get image info
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(val_images_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            continue
        
        # Apply de-raining if model is provided
        if derain_model is not None and use_drsformer:
            # Use DRSformer de-raining with tiling
            from PIL import Image
            try:
                from .drsformer_utils import derain_image as drsformer_derain
            except ImportError:
                from drsformer_utils import derain_image as drsformer_derain
            
            image = Image.open(img_path).convert("RGB")
            image = drsformer_derain(image, derain_model, derain_device, 
                                    tile=drsformer_tile, tile_overlap=drsformer_tile_overlap)
            
            # Run inference on de-rained image
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            target_sizes = torch.tensor([image.size[::-1]]).to(device)
            results = processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=threshold
            )[0]
        else:
            # Use SPDNet or no de-raining
            results, _ = run_inference(img_path, model, processor, device, 
                                       threshold=threshold,
                                       derain_model=derain_model,
                                       derain_device=derain_device)
        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        labels = results['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            total_preds += 1
            lab = int(label)
            
            if lab < 0 or lab >= len(label_to_coco_id):
                skipped_count += 1
                continue
            
            coco_cat_id = label_to_coco_id[lab]
            if coco_cat_id is None:
                skipped_count += 1
                continue
            
            x_min, y_min, x_max, y_max = box.tolist()
            width = float(x_max - x_min)
            height = float(y_max - y_min)
            
            predictions_mapped.append({
                "image_id": int(img_id),
                "category_id": int(coco_cat_id),
                "bbox": [float(x_min), float(y_min), width, height],
                "score": float(score)
            })
    
    print(f"\nTotal raw predictions: {total_preds}")
    print(f"Predictions mapped: {len(predictions_mapped)}")
    print(f"Predictions skipped: {skipped_count}")
    
    return predictions_mapped


def evaluate_coco(predictions, coco_gt, output_file="predictions_mapped.json"):
    """
    Evaluate predictions using COCO metrics.
    
    Args:
        predictions: List of predictions in COCO format
        coco_gt: COCO ground truth object
        output_file: Path to save predictions JSON file
        
    Returns:
        COCOeval object with results
    """
    # Save predictions
    with open(output_file, "w") as f:
        json.dump(predictions, f)
    print(f"Saved mapped predictions to {output_file}")
    
    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(output_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Print metrics
    print("\n" + "="*50)
    print("COCO Evaluation Metrics:")
    print("="*50)
    print(f"mAP @ IoU=0.50:0.95: {coco_eval.stats[0]:.4f}")
    print(f"mAP @ IoU=0.50:      {coco_eval.stats[1]:.4f}")
    print(f"mAP @ IoU=0.75:      {coco_eval.stats[2]:.4f}")
    print(f"mAP (small):         {coco_eval.stats[3]:.4f}")
    print(f"mAP (medium):        {coco_eval.stats[4]:.4f}")
    print(f"mAP (large):         {coco_eval.stats[5]:.4f}")
    
    return coco_eval


def calculate_pr_curve_per_class(coco_gt, coco_dt, iou_threshold=0.5):
    """
    Calculate precision-recall curve for each class.
    
    Args:
        coco_gt: COCO ground truth object
        coco_dt: COCO detection results object
        iou_threshold: IoU threshold for evaluation
        
    Returns:
        Dictionary mapping category IDs to PR curve data
    """
    pr_curves = {}
    category_ids = coco_gt.getCatIds()
    
    for cat_id in tqdm(category_ids, desc="Calculating PR curves"):
        ann_ids = coco_gt.getAnnIds(catIds=[cat_id])
        if len(ann_ids) == 0:
            continue
        
        # Evaluate for this specific category
        coco_eval_cat = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval_cat.params.catIds = [cat_id]
        coco_eval_cat.params.iouThrs = [iou_threshold]
        coco_eval_cat.evaluate()
        coco_eval_cat.accumulate()
        
        precision_all = coco_eval_cat.eval.get('precision', None)
        if precision_all is None:
            continue
        
        try:
            precision = precision_all[0, :, 0, 0, -1]
        except Exception:
            precision = np.squeeze(precision_all)
            if precision.ndim > 1:
                precision = precision[:, 0, -1] if precision.shape[1] > 0 else None
        
        if precision is None:
            continue
        
        recall = coco_eval_cat.params.recThrs
        valid_idx = precision > -1
        
        if not np.any(valid_idx):
            continue
        
        precision = precision[valid_idx]
        recall = np.array(recall)[valid_idx]
        
        if len(precision) == 0:
            continue
        
        pr_curves[cat_id] = {
            'precision': precision,
            'recall': recall,
            'ap': float(np.mean(precision))
        }
    
    return pr_curves


def plot_pr_curves(pr_curves, coco_gt, save_path='pr_curves_top9.png'):
    """
    Plot precision-recall curves for top classes.
    
    Args:
        pr_curves: Dictionary mapping category IDs to PR curve data
        coco_gt: COCO ground truth object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    sorted_classes = sorted(pr_curves.items(), key=lambda x: x[1]['ap'], reverse=True)
    
    print("\nTop 9 classes by AP:")
    for idx, (cat_id, pr_data) in enumerate(sorted_classes[:9]):
        cat_name = coco_gt.loadCats(cat_id)[0]['name']
        ap = pr_data['ap']
        
        print(f"{idx+1}. {cat_name}: AP = {ap:.4f}")
        
        axes[idx].plot(pr_data['recall'], pr_data['precision'], 'b-', linewidth=2)
        axes[idx].set_xlabel('Recall', fontsize=10)
        axes[idx].set_ylabel('Precision', fontsize=10)
        axes[idx].set_title(f'{cat_name}\nAP={ap:.3f}', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([0, 1])
        axes[idx].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nPR curves saved to '{save_path}'")


def plot_overall_pr_curve(pr_curves, save_path='pr_curve_overall.png'):
    """
    Plot overall precision-recall curve across all classes.
    
    Args:
        pr_curves: Dictionary mapping category IDs to PR curve data
        save_path: Path to save the plot
    """
    all_precisions = []
    all_recalls = []
    
    for cat_id, pr_data in pr_curves.items():
        all_precisions.append(pr_data['precision'])
        all_recalls.append(pr_data['recall'])
    
    # Interpolate to common recall points
    recall_levels = np.linspace(0, 1, 101)
    interpolated_precisions = []
    
    for precision, recall in zip(all_precisions, all_recalls):
        interp_precision = np.interp(recall_levels, recall, precision, left=0, right=0)
        interpolated_precisions.append(interp_precision)
    
    mean_precision = np.mean(interpolated_precisions, axis=0)
    overall_ap = np.mean(mean_precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall_levels, mean_precision, 'b-', linewidth=3, label=f'Mean AP = {overall_ap:.3f}')
    
    # Plot individual class curves in background
    for precision, recall in zip(all_precisions[:20], all_recalls[:20]):
        plt.plot(recall, precision, 'gray', alpha=0.2, linewidth=1)
    
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title('Overall Precision-Recall Curve\n(Mean across all classes)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nOverall PR curve saved to '{save_path}'")


def visualize_predictions_comparison(model, processor, device, coco_gt, 
                                    image_ids, val_images_dir, class_names, 
                                    num_samples=4, threshold=0.3,
                                    derain_model=None, derain_device=None):
    """
    Visualize model predictions vs ground truth for sample images.
    
    Args:
        model: RT-DETR model
        processor: RTDetrImageProcessor
        device: Device to run inference on
        coco_gt: COCO ground truth object
        image_ids: List of COCO image IDs
        val_images_dir: Directory containing validation images
        class_names: List of class names
        num_samples: Number of samples to visualize
        threshold: Confidence threshold for predictions
        derain_model: Optional SPDNet model for de-raining (default: None)
        derain_device: Device for de-raining model (default: None, uses device)
    """
    import random
    
    sample_image_ids = random.sample(image_ids, min(num_samples, len(image_ids)))
    
    fig, axes = plt.subplots(len(sample_image_ids), 2, figsize=(14, 4 * len(sample_image_ids)))
    if len(sample_image_ids) == 1:
        axes = np.expand_dims(axes, 0)
    axes = axes.reshape(-1, 2)
    
    for row_idx, img_id in enumerate(sample_image_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(val_images_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"Missing image: {img_path}, skipping")
            continue
        
        image = Image.open(img_path).convert("RGB")
        
        # Ground truth (left)
        ax_gt = axes[row_idx, 0]
        ax_gt.imshow(image)
        ax_gt.set_title(f"GT - {img_info['file_name']}")
        ax_gt.axis("off")
        
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        for ann in anns:
            x, y, w, h = ann['bbox']
            cat_name = coco_gt.loadCats(ann['category_id'])[0]['name']
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
            ax_gt.add_patch(rect)
            ax_gt.text(x, y - 2, cat_name, color='lime', fontsize=8, backgroundcolor='black', alpha=0.7)
        
        # Predictions (right)
        ax_pred = axes[row_idx, 1]
        ax_pred.imshow(image)
        ax_pred.set_title("Predictions")
        ax_pred.axis("off")
        
        results_pred, _ = run_inference(img_path, model, processor, device, 
                                        threshold=threshold,
                                        derain_model=derain_model,
                                        derain_device=derain_device)
        boxes = results_pred['boxes'].cpu().numpy()
        scores = results_pred['scores'].cpu().numpy()
        labels = results_pred['labels'].cpu().numpy()
        
        keep_idx = np.where(scores >= threshold)[0]
        for idx in keep_idx[:50]:
            x_min, y_min, x_max, y_max = boxes[idx]
            w, h = x_max - x_min, y_max - y_min
            label = class_names[labels[idx]] if labels[idx] < len(class_names) else str(int(labels[idx]))
            score = scores[idx]
            rect = Rectangle((x_min, y_min), w, h, linewidth=2, edgecolor='red', facecolor='none')
            ax_pred.add_patch(rect)
            ax_pred.text(x_min, y_min - 2, f"{label} {score:.2f}", color='red', fontsize=8, backgroundcolor='black', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
