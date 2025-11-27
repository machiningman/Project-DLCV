#!/usr/bin/env python3
"""
MixedRain mAP Evaluation - Focused Comparison

Compares three models using COCO mAP metrics:
1. Vanilla RT-DETR (baseline)
2. Always SPDNet → RT-DETR (de-rain all images)
3. Conditional Pipeline (Rain Detector → SPDNet → RT-DETR)

Usage:
    python Eval_MixedRain_mAP.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json
import tempfile
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import cv2
cv2.setNumThreads(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import pycocotools for mAP
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_PYCOCOTOOLS = True
except ImportError:
    print("[ERROR] pycocotools required. Install with: pip install pycocotools")
    HAS_PYCOCOTOOLS = False

# =============================================================================
# Configuration
# =============================================================================

MIXEDRAIN_DIR = "E:/Python/DLCV/Project DLCV/mixedrain"
LABELS_FILE = os.path.join(MIXEDRAIN_DIR, "labels.txt")

# Model paths
ORIGINAL_SPDNET_PATH = "E:/Python/DLCV/Project DLCV/model_spa.pt"
ADAPTED_SPDNET_PATH = "./outputs_spdnet_feature_adaptation/spdnet_adapted_best.pt"
RAIN_DETECTOR_PATH = "./rain_detector_pretrained/rain_detector_best.pt"
RTDETR_MODEL_NAME = "PekingU/rtdetr_r18vd"
NUM_LABELS = 80

# Settings
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.01  # Low threshold for mAP calculation
RAIN_THRESHOLD = 0.3  # Lower threshold to catch more degraded images

# SPDNet Configuration
SPDNET_N_FEATS = 32
SPDNET_N_RESBLOCKS = 3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Dataset
# =============================================================================

class MixedRainDataset(torch.utils.data.Dataset):
    """MixedRain dataset with YOLO format labels"""
    
    def __init__(self, data_dir, labels_file, image_size=640):
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Parse labels
        self.annotations = defaultdict(list)
        self.image_ids = []
        
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        img_id = parts[0]
                        class_id = int(parts[1])
                        x_center = float(parts[2])
                        y_center = float(parts[3])
                        width = float(parts[4])
                        height = float(parts[5])
                        
                        self.annotations[img_id].append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
                        
                        if img_id not in self.image_ids:
                            self.image_ids.append(img_id)
        
        # Find image files
        self.image_files = {}
        for img_id in self.image_ids:
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = os.path.join(data_dir, f"{img_id}{ext}")
                if os.path.exists(img_path):
                    self.image_files[img_id] = img_path
                    break
        
        self.image_ids = [img_id for img_id in self.image_ids if img_id in self.image_files]
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        
        print(f"[Dataset] Loaded {len(self.image_ids)} images")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.image_files[img_id]
        
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size  # (W, H)
        
        image_tensor = self.transform(image)
        
        # Convert YOLO to COCO format [x, y, w, h]
        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []
        areas = []
        
        orig_w, orig_h = orig_size
        
        for ann in anns:
            x_center = ann['x_center']
            y_center = ann['y_center']
            w = ann['width']
            h = ann['height']
            
            # Convert to absolute COCO format [x, y, width, height]
            x = (x_center - w / 2) * orig_w
            y = (y_center - h / 2) * orig_h
            box_w = w * orig_w
            box_h = h * orig_h
            
            if box_w > 0 and box_h > 0:
                boxes.append([x, y, box_w, box_h])
                labels.append(ann['class_id'])
                areas.append(box_w * box_h)
        
        return {
            'image': image_tensor,
            'boxes': boxes,
            'labels': labels,
            'areas': areas,
            'image_id': idx,  # Numeric ID for COCO format
            'orig_size': orig_size,
            'str_id': img_id
        }


def collate_fn(batch):
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'boxes': [item['boxes'] for item in batch],
        'labels': [item['labels'] for item in batch],
        'areas': [item['areas'] for item in batch],
        'image_ids': [item['image_id'] for item in batch],
        'orig_sizes': [item['orig_size'] for item in batch],
        'str_ids': [item['str_id'] for item in batch]
    }


# =============================================================================
# Model Loading
# =============================================================================

def load_models():
    """Load all required models"""
    from utils.model_utils import load_model_and_processor
    from utils.spdnet_utils import load_spdnet_model
    from utils.rain_detector import load_rain_detector
    
    print("\n" + "="*60)
    print("Loading Models")
    print("="*60)
    
    # RT-DETR
    print("[1/4] Loading RT-DETR...")
    model, processor = load_model_and_processor(RTDETR_MODEL_NAME, num_labels=NUM_LABELS)
    model = model.to(DEVICE)
    model.eval()
    print(f"      RT-DETR loaded on {DEVICE}")
    
    # Original SPDNet
    print("[2/4] Loading Original SPDNet...")
    original_spdnet = load_spdnet_model(
        ORIGINAL_SPDNET_PATH,
        device=DEVICE,
        n_feats=SPDNET_N_FEATS,
        n_resblocks=SPDNET_N_RESBLOCKS
    )
    original_spdnet.eval()
    print(f"      Original SPDNet loaded")
    
    # Adapted SPDNet
    print("[3/4] Loading Adapted SPDNet...")
    adapted_spdnet = load_spdnet_model(
        ADAPTED_SPDNET_PATH,
        device=DEVICE,
        n_feats=SPDNET_N_FEATS,
        n_resblocks=SPDNET_N_RESBLOCKS
    )
    adapted_spdnet.eval()
    print(f"      Adapted SPDNet loaded")
    
    # Rain Detector
    print("[4/4] Loading Rain Detector...")
    rain_detector = load_rain_detector(RAIN_DETECTOR_PATH, device=DEVICE)
    rain_detector.eval()
    print(f"      Rain Detector loaded")
    
    return model, processor, original_spdnet, adapted_spdnet, rain_detector


# =============================================================================
# Helpers
# =============================================================================

def apply_spdnet(spdnet, images):
    """Apply SPDNet de-raining"""
    spdnet_input = images * 255.0
    with torch.no_grad():
        output = spdnet(spdnet_input)
        if isinstance(output, tuple):
            output = output[0]
    return torch.clamp(output / 255.0, 0, 1)


def apply_rain_detector(rain_detector, images, threshold=RAIN_THRESHOLD):
    """Apply rain detector"""
    import torch.nn.functional as F
    
    resized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)
    normalized = (resized - mean) / std
    
    with torch.no_grad():
        logits = rain_detector(normalized)
        scores = torch.sigmoid(logits)
    
    return scores, scores > threshold


def run_detection(model, processor, images, orig_sizes, device, conf_thresh=CONFIDENCE_THRESHOLD):
    """Run RT-DETR and return results"""
    with torch.no_grad():
        outputs = model(pixel_values=images)
    
    # Post-process to original sizes
    target_sizes = torch.tensor([(h, w) for w, h in orig_sizes]).to(device)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=conf_thresh
    )
    
    return results


# =============================================================================
# COCO mAP Evaluation
# =============================================================================

def create_coco_gt(dataset):
    """Create COCO ground truth object"""
    gt_dict = {
        'info': {'description': 'MixedRain Dataset'},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [{'id': i, 'name': f'class_{i}'} for i in range(80)]
    }
    
    ann_id = 1
    for idx in range(len(dataset)):
        sample = dataset[idx]
        img_id = sample['image_id']
        orig_w, orig_h = sample['orig_size']
        
        gt_dict['images'].append({
            'id': img_id,
            'width': orig_w,
            'height': orig_h,
            'file_name': f'{sample["str_id"]}.jpg'
        })
        
        for box, label, area in zip(sample['boxes'], sample['labels'], sample['areas']):
            gt_dict['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': label,
                'bbox': box,  # [x, y, w, h]
                'area': area,
                'iscrowd': 0
            })
            ann_id += 1
    
    # Save to temp file and load with pycocotools
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(gt_dict, f)
        temp_path = f.name
    
    coco_gt = COCO(temp_path)
    os.unlink(temp_path)
    
    return coco_gt


def evaluate_with_coco(predictions, coco_gt):
    """Evaluate predictions using COCO metrics"""
    if len(predictions) == 0:
        return {'mAP': 0.0, 'AP50': 0.0, 'AP75': 0.0}
    
    # Save predictions to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(predictions, f)
        temp_path = f.name
    
    coco_dt = coco_gt.loadRes(temp_path)
    os.unlink(temp_path)
    
    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return {
        'mAP': coco_eval.stats[0],    # AP @ IoU=0.50:0.95
        'AP50': coco_eval.stats[1],   # AP @ IoU=0.50
        'AP75': coco_eval.stats[2],   # AP @ IoU=0.75
        'AP_small': coco_eval.stats[3],
        'AP_medium': coco_eval.stats[4],
        'AP_large': coco_eval.stats[5]
    }


def generate_predictions(model, processor, dataloader, device, spdnet=None, 
                         rain_detector=None, conditional=False, desc="Evaluating"):
    """Generate predictions for all images and track inference time"""
    import time
    
    model.eval()
    if spdnet is not None:
        spdnet.eval()
    if rain_detector is not None:
        rain_detector.eval()
    
    predictions = []
    derained_count = 0
    total_count = 0
    
    # Timing stats
    total_time = 0.0
    rain_det_time = 0.0
    derain_time = 0.0
    detect_time = 0.0
    
    # Warmup
    print("  Warmup...")
    warmup_img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    for _ in range(3):
        with torch.no_grad():
            _ = model(pixel_values=warmup_img)
    torch.cuda.synchronize()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            images = batch['images'].to(device)
            image_ids = batch['image_ids']
            orig_sizes = batch['orig_sizes']
            
            batch_size = images.shape[0]
            total_count += batch_size
            
            torch.cuda.synchronize()
            batch_start = time.perf_counter()
            
            # Apply de-raining based on mode
            if spdnet is not None:
                if conditional and rain_detector is not None:
                    # Conditional: only de-rain if detected as rainy
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _, rainy_mask = apply_rain_detector(rain_detector, images)
                    torch.cuda.synchronize()
                    rain_det_time += time.perf_counter() - t0
                    
                    if rainy_mask.any():
                        rainy_idx = rainy_mask.nonzero(as_tuple=True)[0]
                        derained_count += len(rainy_idx)
                        rainy_images = images[rainy_idx]
                        
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        images[rainy_idx] = apply_spdnet(spdnet, rainy_images)
                        torch.cuda.synchronize()
                        derain_time += time.perf_counter() - t0
                else:
                    # Always de-rain
                    derained_count += batch_size
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    images = apply_spdnet(spdnet, images)
                    torch.cuda.synchronize()
                    derain_time += time.perf_counter() - t0
            
            # Run detection
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            results = run_detection(model, processor, images, orig_sizes, device)
            torch.cuda.synchronize()
            detect_time += time.perf_counter() - t0
            
            torch.cuda.synchronize()
            total_time += time.perf_counter() - batch_start
            
            # Convert to COCO format
            for i, result in enumerate(results):
                boxes = result['boxes'].cpu().numpy()
                scores = result['scores'].cpu().numpy()
                labels = result['labels'].cpu().numpy()
                img_id = image_ids[i]
                
                for box, score, label in zip(boxes, scores, labels):
                    # Convert xyxy to xywh
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    
                    predictions.append({
                        'image_id': int(img_id),
                        'category_id': int(label),
                        'bbox': [float(x1), float(y1), float(w), float(h)],
                        'score': float(score)
                    })
    
    # Calculate timing stats
    avg_time_ms = (total_time / total_count) * 1000
    avg_detect_ms = (detect_time / total_count) * 1000
    avg_derain_ms = (derain_time / derained_count) * 1000 if derained_count > 0 else 0
    fps = total_count / total_time
    
    timing_stats = {
        'total_images': total_count,
        'derained_images': derained_count,
        'total_time_s': total_time,
        'avg_time_ms': avg_time_ms,
        'avg_detect_ms': avg_detect_ms,
        'avg_derain_ms': avg_derain_ms,
        'fps': fps
    }
    
    print(f"  De-rained: {derained_count}/{total_count} images")
    print(f"  Avg time per image: {avg_time_ms:.1f} ms ({fps:.1f} FPS)")
    
    return predictions, timing_stats


# =============================================================================
# Main
# =============================================================================

def main():
    if not HAS_PYCOCOTOOLS:
        print("ERROR: pycocotools required for mAP evaluation")
        return
    
    print("\n" + "="*60)
    print("MixedRain mAP Evaluation")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {MIXEDRAIN_DIR}")
    print(f"Rain Detection Threshold: {RAIN_THRESHOLD}")
    
    # Load dataset
    print("\n[Loading Dataset]")
    dataset = MixedRainDataset(MIXEDRAIN_DIR, LABELS_FILE, IMAGE_SIZE)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Create COCO GT
    print("[Creating COCO Ground Truth]")
    coco_gt = create_coco_gt(dataset)
    print(f"  GT annotations: {len(coco_gt.anns)}")
    print(f"  GT images: {len(coco_gt.imgs)}")
    
    # Load models
    model, processor, original_spdnet, adapted_spdnet, rain_detector = load_models()
    
    results = {}
    timing = {}
    
    # 1. Vanilla RT-DETR
    print("\n" + "="*60)
    print("1. Vanilla RT-DETR (No De-raining)")
    print("="*60)
    preds_vanilla, timing['vanilla'] = generate_predictions(
        model, processor, dataloader, DEVICE,
        spdnet=None, rain_detector=None, conditional=False,
        desc="Vanilla RT-DETR"
    )
    results['vanilla'] = evaluate_with_coco(preds_vanilla, coco_gt)
    
    # 2. Original SPDNet → RT-DETR
    print("\n" + "="*60)
    print("2. Original SPDNet → RT-DETR (Always De-rain)")
    print("="*60)
    preds_orig_spdnet, timing['original_spdnet'] = generate_predictions(
        model, processor, dataloader, DEVICE,
        spdnet=original_spdnet, rain_detector=None, conditional=False,
        desc="Original SPDNet"
    )
    results['original_spdnet'] = evaluate_with_coco(preds_orig_spdnet, coco_gt)
    
    # 3. Adapted SPDNet → RT-DETR
    print("\n" + "="*60)
    print("3. Adapted SPDNet → RT-DETR (Always De-rain)")
    print("="*60)
    preds_adapted, timing['adapted_spdnet'] = generate_predictions(
        model, processor, dataloader, DEVICE,
        spdnet=adapted_spdnet, rain_detector=None, conditional=False,
        desc="Adapted SPDNet"
    )
    results['adapted_spdnet'] = evaluate_with_coco(preds_adapted, coco_gt)
    
    # 4. Conditional Pipeline (with Adapted SPDNet)
    print("\n" + "="*60)
    print("4. Conditional Pipeline (Rain Detector → Adapted SPDNet → RT-DETR)")
    print("="*60)
    preds_conditional, timing['conditional'] = generate_predictions(
        model, processor, dataloader, DEVICE,
        spdnet=adapted_spdnet, rain_detector=rain_detector, conditional=True,
        desc="Conditional (Adapted)"
    )
    results['conditional'] = evaluate_with_coco(preds_conditional, coco_gt)
    
    # 5. Conditional Pipeline (with Original SPDNet)
    print("\n" + "="*60)
    print("5. Conditional Pipeline (Rain Detector → Original SPDNet → RT-DETR)")
    print("="*60)
    preds_conditional_orig, timing['conditional_original'] = generate_predictions(
        model, processor, dataloader, DEVICE,
        spdnet=original_spdnet, rain_detector=rain_detector, conditional=True,
        desc="Conditional (Original)"
    )
    results['conditional_original'] = evaluate_with_coco(preds_conditional_orig, coco_gt)
    
    # Print comparison
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    print(f"{'Model':<42} {'mAP':>8} {'AP50':>8} {'AP75':>8} {'ms/img':>8} {'FPS':>8}")
    print("-"*82)
    
    for name, key in [
        ("1. Vanilla RT-DETR", "vanilla"),
        ("2. Original SPDNet → RT-DETR", "original_spdnet"),
        ("3. Adapted SPDNet → RT-DETR", "adapted_spdnet"),
        ("4. Conditional (Adapted SPDNet)", "conditional"),
        ("5. Conditional (Original SPDNet)", "conditional_original")
    ]:
        m = results[key]
        t = timing[key]
        print(f"{name:<42} {m['mAP']*100:>7.2f}% {m['AP50']*100:>7.2f}% {m['AP75']*100:>7.2f}% {t['avg_time_ms']:>7.1f} {t['fps']:>7.1f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Best mAP
    best_key = max(results, key=lambda k: results[k]['mAP'])
    print(f"✓ Best mAP: {best_key} ({results[best_key]['mAP']*100:.2f}%)")
    
    # Fastest
    fastest_key = max(timing, key=lambda k: timing[k]['fps'])
    print(f"✓ Fastest: {fastest_key} ({timing[fastest_key]['fps']:.1f} FPS)")
    
    # Speed comparison
    vanilla_fps = timing['vanilla']['fps']
    cond_fps = timing['conditional']['fps']
    orig_fps = timing['original_spdnet']['fps']
    print(f"\nSpeed Analysis:")
    print(f"  - Conditional vs Original SPDNet: {cond_fps/orig_fps:.1f}x faster")
    print(f"  - Conditional vs Vanilla: {cond_fps/vanilla_fps:.1f}x slower")
    
    # mAP Analysis
    print(f"\nmAP Analysis:")
    vanilla_map = results['vanilla']['mAP']
    for name, key in [
        ("Original SPDNet (Always)", "original_spdnet"),
        ("Adapted SPDNet (Always)", "adapted_spdnet"),
        ("Conditional (Adapted)", "conditional"),
        ("Conditional (Original)", "conditional_original")
    ]:
        diff = (results[key]['mAP'] - vanilla_map) * 100
        sign = "+" if diff >= 0 else ""
        print(f"  - {name} vs Vanilla: {sign}{diff:.2f}%")
    
    # Save results
    output_path = os.path.join(MIXEDRAIN_DIR, "mAP_results.json")
    full_results = {
        'metrics': results,
        'timing': timing
    }
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results, timing


if __name__ == "__main__":
    main()
