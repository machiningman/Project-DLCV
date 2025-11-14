#!/usr/bin/env python3
"""
Evaluation script for RT-DETR object detection model.

This script loads a trained model and evaluates it on COCO validation dataset.
Compares three methods:
1. Vanilla RT-DETR (no de-raining)
2. SPDNet + RT-DETR
3. DRSformer + RT-DETR
"""

import os
import torch
import warnings
from pycocotools.coco import COCO

# Import utility modules
from utils.model_utils import load_model_and_processor
from utils.eval_utils import (
    COCO_CLASS_NAMES,
    build_label_mapping,
    generate_predictions,
    evaluate_coco,
    calculate_pr_curve_per_class,
    plot_pr_curves,
    plot_overall_pr_curve,
    visualize_predictions_comparison
)

warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = "E:\\Python\\DLCV\\Project\\dataset\\coco_rain"
# MODEL_PATH = "E:\\Python\\DLCV\\Project\\outputs\\epoch-12_0.1dataset"
MODEL_PATH = "PekingU/rtdetr_r18vd"  # Use pretrained model

# De-raining Model Paths
SPDNET_MODEL_PATH = "E:\\Python\\DLCV\\Project\\model_spa.pt"
DRSFORMER_MODEL_PATH = "E:\\Python\\DLCV\\DRSformer\\pretrained_models\\deraining.pth"

# SPDNet Configuration
SPDNET_N_FEATS = 32  # Number of feature maps in SPDNet
SPDNET_N_RESBLOCKS = 3  # Number of residual blocks in SPDNet

# DRSformer Configuration
DRSFORMER_TILE = 256  # Tile size for DRSformer (e.g., 512, 720, None for full image)
                      # Using tiles significantly speeds up inference!
DRSFORMER_TILE_OVERLAP = 32  # Overlap between tiles

# Dataset sampling
DATASET_FRACTION = 1.0  # Use 10% of dataset for faster evaluation (set to 1.0 for full dataset)

CONFIDENCE_THRESHOLD = 0.3
INFERENCE_THRESHOLD = 0.01  # Low threshold for COCO eval to preserve score distribution

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    """Main evaluation function - compares Vanilla, SPDNet, and DRSformer"""
    print("=" * 80)
    print("RT-DETR De-raining Comparison Evaluation")
    print("=" * 80)
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Device: {DEVICE}")
    print(f"\nComparing three methods:")
    print("  1. Vanilla RT-DETR (no de-raining)")
    print("  2. SPDNet + RT-DETR")
    print(f"  3. DRSformer + RT-DETR (tile={DRSFORMER_TILE})")
    if DRSFORMER_TILE:
        print(f"     → Using tiled inference for speed optimization")
    
    # Load de-raining models
    print("\n" + "=" * 80)
    print("Loading De-raining Models")
    print("=" * 80)
    
    # Load SPDNet
    spdnet_model = None
    if os.path.exists(SPDNET_MODEL_PATH):
        print("\nLoading SPDNet model...")
        from utils.spdnet_utils import load_spdnet_model
        try:
            spdnet_model = load_spdnet_model(
                SPDNET_MODEL_PATH, 
                device=DEVICE,
                n_feats=SPDNET_N_FEATS,
                n_resblocks=SPDNET_N_RESBLOCKS
            )
            print(f"✓ SPDNet model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading SPDNet model: {e}")
    else:
        print(f"\n✗ SPDNet model not found at: {SPDNET_MODEL_PATH}")
    
    # Load DRSformer
    drsformer_model = None
    if os.path.exists(DRSFORMER_MODEL_PATH):
        print("\nLoading DRSformer model...")
        from utils.drsformer_utils import load_drsformer_model
        try:
            drsformer_model = load_drsformer_model(DRSFORMER_MODEL_PATH, device=DEVICE)
            print(f"✓ DRSformer model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading DRSformer model: {e}")
    else:
        print(f"\n✗ DRSformer model not found at: {DRSFORMER_MODEL_PATH}")
    
    if spdnet_model is None and drsformer_model is None:
        print("\n⚠ WARNING: No de-raining models loaded. Will only evaluate vanilla RT-DETR.")
    
    # Load RT-DETR model and processor
    print("\n" + "=" * 80)
    print("Loading RT-DETR Model")
    print("=" * 80)
    model, processor = load_model_and_processor(model_name=MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print(f"✓ RT-DETR model loaded on device: {DEVICE}")
    
    # Load COCO dataset
    print("\n" + "=" * 80)
    print("Loading COCO Validation Dataset")
    print("=" * 80)
    val_annotations_file = os.path.join(DATASET_PATH, "annotations", "instances_val2017.json")
    val_images_dir = os.path.join(DATASET_PATH, "val2017")
    
    print(f"Annotations: {val_annotations_file}")
    print(f"Images: {val_images_dir}")
    
    coco_gt = COCO(val_annotations_file)
    all_image_ids = coco_gt.getImgIds()
    
    # Sample subset of images if requested
    if DATASET_FRACTION < 1.0:
        import random
        random.seed(42)  # For reproducibility
        num_images = int(len(all_image_ids) * DATASET_FRACTION)
        image_ids = random.sample(all_image_ids, num_images)
        print(f"✓ Using {len(image_ids)} images ({DATASET_FRACTION*100:.1f}% of {len(all_image_ids)} total)")
    else:
        image_ids = all_image_ids
        print(f"✓ Using all {len(image_ids)} validation images")
    
    # Build label mapping
    print("\n" + "=" * 80)
    print("Building Label Mapping")
    print("=" * 80)
    label_to_coco_id = build_label_mapping(COCO_CLASS_NAMES, coco_gt)
    print(f"✓ Loaded {len(label_to_coco_id)} label mappings")
    
    # Store results for all methods
    all_results = {}
    
    # ==================== Method 1: Vanilla RT-DETR ====================
    print("\n" + "=" * 80)
    print("METHOD 1: Vanilla RT-DETR (No De-raining)")
    print("=" * 80)
    print("Generating predictions...")
    predictions_vanilla = generate_predictions(
        model=model,
        processor=processor,
        device=DEVICE,
        coco_gt=coco_gt,
        image_ids=image_ids,
        val_images_dir=val_images_dir,
        label_to_coco_id=label_to_coco_id,
        class_names=COCO_CLASS_NAMES,
        threshold=INFERENCE_THRESHOLD,
        derain_model=None,  # No de-raining
        derain_device=None
    )
    
    print("Evaluating with COCO metrics...")
    coco_eval_vanilla = evaluate_coco(
        predictions=predictions_vanilla,
        coco_gt=coco_gt,
        output_file="predictions_vanilla.json"
    )
    all_results['vanilla'] = coco_eval_vanilla.stats
    
    # ==================== Method 2: SPDNet + RT-DETR ====================
    if spdnet_model is not None:
        print("\n" + "=" * 80)
        print("METHOD 2: SPDNet + RT-DETR")
        print("=" * 80)
        print("Generating predictions with SPDNet de-raining...")
        predictions_spdnet = generate_predictions(
            model=model,
            processor=processor,
            device=DEVICE,
            coco_gt=coco_gt,
            image_ids=image_ids,
            val_images_dir=val_images_dir,
            label_to_coco_id=label_to_coco_id,
            class_names=COCO_CLASS_NAMES,
            threshold=INFERENCE_THRESHOLD,
            derain_model=spdnet_model,
            derain_device=DEVICE
        )
        
        print("Evaluating with COCO metrics...")
        coco_eval_spdnet = evaluate_coco(
            predictions=predictions_spdnet,
            coco_gt=coco_gt,
            output_file="predictions_spdnet.json"
        )
        all_results['spdnet'] = coco_eval_spdnet.stats
    
    # ==================== Method 3: DRSformer + RT-DETR ====================
    if drsformer_model is not None:
        print("\n" + "=" * 80)
        print(f"METHOD 3: DRSformer + RT-DETR (tile={DRSFORMER_TILE})")
        print("=" * 80)
        print("Generating predictions with DRSformer de-raining...")
        if DRSFORMER_TILE:
            print(f"  Using tiled inference: tile_size={DRSFORMER_TILE}, overlap={DRSFORMER_TILE_OVERLAP}")
        
        # Note: DRSformer uses a different derain function
        from utils.drsformer_utils import derain_image as drsformer_derain
        
        # Use wrapper approach with tile parameters
        predictions_drsformer = generate_predictions(
            model=model,
            processor=processor,
            device=DEVICE,
            coco_gt=coco_gt,
            image_ids=image_ids,
            val_images_dir=val_images_dir,
            label_to_coco_id=label_to_coco_id,
            class_names=COCO_CLASS_NAMES,
            threshold=INFERENCE_THRESHOLD,
            derain_model=drsformer_model,
            derain_device=DEVICE,
            use_drsformer=True,  # Flag to use DRSformer derain function
            drsformer_tile=DRSFORMER_TILE,
            drsformer_tile_overlap=DRSFORMER_TILE_OVERLAP
        )
        
        print("Evaluating with COCO metrics...")
        coco_eval_drsformer = evaluate_coco(
            predictions=predictions_drsformer,
            coco_gt=coco_gt,
            output_file="predictions_drsformer.json"
        )
        all_results['drsformer'] = coco_eval_drsformer.stats
    
    # # ==================== Comparison Summary ====================
    # print("\n" + "=" * 80)
    # print("EVALUATION COMPARISON SUMMARY")
    # print("=" * 80)
    
    # # Print comparison table
    # print("\n" + "-" * 100)
    # print(f"{'Metric':<30} {'Vanilla RT-DETR':<20} {'SPDNet + RT-DETR':<20} {'DRSformer + RT-DETR':<20}")
    # print("-" * 100)
    
    # metric_names = [
    #     "mAP @ IoU=0.50:0.95",
    #     "mAP @ IoU=0.50",
    #     "mAP @ IoU=0.75",
    #     "mAP (small)",
    #     "mAP (medium)",
    #     "mAP (large)",
    #     "mAR @ maxDets=1",
    #     "mAR @ maxDets=10",
    #     "mAR @ maxDets=100",
    #     "mAR (small)",
    #     "mAR (medium)",
    #     "mAR (large)"
    # ]
    
    # for i, metric_name in enumerate(metric_names):
    #     vanilla_val = all_results['vanilla'][i]
    #     spdnet_val = all_results.get('spdnet', [0]*12)[i] if spdnet_model else 0
    #     drsformer_val = all_results.get('drsformer', [0]*12)[i] if drsformer_model else 0
        
    #     vanilla_str = f"{vanilla_val:.4f}"
    #     spdnet_str = f"{spdnet_val:.4f}" if spdnet_model else "N/A"
    #     drsformer_str = f"{drsformer_val:.4f}" if drsformer_model else "N/A"
        
    #     # Add improvement indicators
    #     if spdnet_model and spdnet_val > vanilla_val:
    #         spdnet_str += " ↑"
    #     if drsformer_model and drsformer_val > vanilla_val:
    #         drsformer_str += " ↑"
        
    #     print(f"{metric_name:<30} {vanilla_str:<20} {spdnet_str:<20} {drsformer_str:<20}")
    
    # print("-" * 100)
    
    # # Determine best method
    # print("\n" + "=" * 80)
    # print("RECOMMENDATION")
    # print("=" * 80)
    
    # best_method = "Vanilla RT-DETR"
    # best_map = all_results['vanilla'][0]
    
    # if spdnet_model and all_results.get('spdnet', [0])[0] > best_map:
    #     best_method = "SPDNet + RT-DETR"
    #     best_map = all_results['spdnet'][0]
    
    # if drsformer_model and all_results.get('drsformer', [0])[0] > best_map:
    #     best_method = "DRSformer + RT-DETR"
    #     best_map = all_results['drsformer'][0]
    
    # print(f"\n✓ Best performing method: {best_method}")
    # print(f"  mAP @ IoU=0.50:0.95: {best_map:.4f}")
    
    # if best_method != "Vanilla RT-DETR":
    #     improvement = (best_map - all_results['vanilla'][0]) * 100
    #     print(f"  Improvement over Vanilla: +{improvement:.2f}%")
    
    # print("\n" + "=" * 80)
    # print("EVALUATION COMPLETED SUCCESSFULLY!")
    # print("=" * 80)


if __name__ == "__main__":
    main()
